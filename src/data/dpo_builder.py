import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from src.data.preprocessor import parse_problems_column, add_choices_len
from src.prompt.prompt_builder import PromptBuilder, PromptConfig
from src.training.model_loader import ModelConfig, load_model_inference
from src.utils.seed import set_seed


def main(
    config_path: str,
    adapter_path: str = None,
    output_dir: str = None,
    data_path: str = None,
    valid_ratio: float = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = Path(config_path).parent
    adapter_path = adapter_path or str(project_root / cfg["dpo"]["sft_adapter_path"])
    output_dir = output_dir or str(project_root / "data/dpo_outputs")
    data_path = data_path or str(project_root / cfg["data"]["train_path"])
    valid_ratio = valid_ratio or cfg["data"]["valid_ratio"]
    seed = cfg["data"]["seed"]
    max_new_tokens = cfg["inference"].get("max_new_tokens", 30)

    set_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DPO Builder - Generate inference results for DPO training")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"SFT Adapter: {adapter_path}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Valid Ratio: {valid_ratio}")
    print(f"Max New Tokens: {max_new_tokens}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = parse_problems_column(df)
    df = add_choices_len(df)
    print(f"Loaded {len(df)} rows")

    print(f"\nSplitting data (valid_ratio={valid_ratio}, seed={seed})...")
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_ratio,
        stratify=df["choices_len"],
        random_state=seed,
    )
    print(f"Train: {len(train_df)} rows")
    print(f"Valid: {len(valid_df)} rows")

    print("\nCreating model config...")
    model_cfg = ModelConfig(
        model_name_or_path=cfg["model"]["model_name_or_path"],
        use_4bit=cfg["model"]["use_4bit"],
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["model"]["bnb_4bit_use_double_quant"],
        compute_dtype=cfg["model"]["compute_dtype"],
        device_map=cfg["model"]["device_map"],
        use_gradient_checkpointing=False,  # inference 시 False
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )

    print(f"\nLoading tokenizer from {model_cfg.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading model from {adapter_path}...")
    model = load_model_inference(model_cfg, adapter_path)
    model.eval()
    print("Model loaded successfully!")

    prompt_cfg = PromptConfig(
        policy=cfg["prompt"]["policy"],
        mode="test",
        verbose=False
    )
    builder = PromptBuilder(prompt_cfg)
    print("PromptBuilder ready!")

    print("\n" + "=" * 80)
    print("Running inference on TRAIN set")
    print("=" * 80)
    train_gen_df = process_dataset(
        df=train_df,
        builder=builder,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        desc="Train Generation",
    )

    train_acc = train_gen_df['is_correct'].mean()
    print(f"\nTrain Accuracy: {train_acc:.4f} ({train_gen_df['is_correct'].sum()}/{len(train_gen_df)})")

    print("\n" + "=" * 80)
    print("Running inference on VALID set")
    print("=" * 80)
    valid_gen_df = process_dataset(
        df=valid_df,
        builder=builder,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        desc="Valid Generation",
    )

    valid_acc = valid_gen_df['is_correct'].mean()
    print(f"\nValid Accuracy: {valid_acc:.4f} ({valid_gen_df['is_correct'].sum()}/{len(valid_gen_df)})")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Valid Accuracy: {valid_acc:.4f}")
    print(f"Generalization Gap: {(train_acc - valid_acc):.4f}")

    train_output_path = output_dir / "train_gen.csv"
    valid_output_path = output_dir / "valid_gen.csv"

    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)

    train_gen_df_save = train_gen_df.copy()
    valid_gen_df_save = valid_gen_df.copy()

    for col in ["top5_candidates", "digit_probs_1_to_k"]:
        train_gen_df_save[col] = train_gen_df_save[col].astype(str)
        valid_gen_df_save[col] = valid_gen_df_save[col].astype(str)

    train_gen_df_save.to_csv(train_output_path, index=False)
    valid_gen_df_save.to_csv(valid_output_path, index=False)

    print(f"Saved train results to {train_output_path}")
    print(f"Saved valid results to {valid_output_path}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

    return train_gen_df, valid_gen_df

def extract_answer(text: str, k: int) -> str:
    numbers = re.findall(rf'[1-{k}]', str(text))
    return numbers[-1] if numbers else "no"


def digit_only_probs_and_margin(step_logits: torch.Tensor, tokenizer, k: int) -> Dict[str, Any]:
    digit_tokens = [str(i) for i in range(1, k + 1)]
    digit_token_ids = []

    for digit in digit_tokens:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) == 1:
            digit_token_ids.append(encoded[0])
        else:
            digit_token_ids.append(encoded[0])

    digit_logits = torch.tensor([step_logits[tid].item() for tid in digit_token_ids])
    digit_probs = torch.softmax(digit_logits, dim=-1)

    top2_values, top2_indices = torch.topk(digit_probs, k=min(2, k))

    digit_top1 = str(top2_indices[0].item() + 1)  # 1-indexed
    digit_top2 = str(top2_indices[1].item() + 1) if k >= 2 else "N/A"

    if k >= 2:
        digit_margin = (top2_values[0] - top2_values[1]).item()
    else:
        digit_margin = 0.0

    return {
        "digit_probs": digit_probs.tolist(),
        "digit_margin": digit_margin,
        "digit_top1": digit_top1,
        "digit_top2": digit_top2,
    }


def generate_for_row_with_top5(
    row_dict: Dict[str, Any],
    builder: PromptBuilder,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: str,
    max_new_tokens: int = 30,
) -> Dict[str, Any]:
    output = builder.build_message(row_dict)
    messages = output["messages"]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)

    k = int(row_dict["choices_len"])
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 끝에서 2번째 step의 logits 사용 (답변 digit이 나오는 위치)
    step_logits = outputs.scores[-2][0]

    top5_values, top5_indices = torch.topk(step_logits, k=5)
    probs_full = torch.softmax(step_logits, dim=-1)
    top5_candidates = []
    for rank, (logit_val, token_id) in enumerate(zip(top5_values, top5_indices)):
        top5_candidates.append({
            "rank": rank + 1,
            "token_id": token_id.item(),
            "token": tokenizer.decode([token_id.item()]),
            "logit": logit_val.item(),
            "prob_full_vocab": probs_full[token_id].item(),
        })

    digit_info = digit_only_probs_and_margin(step_logits, tokenizer, k)
    digit_margin = digit_info["digit_margin"]
    digit_probs = digit_info["digit_probs"]

    predicted_answer = extract_answer(generated_text, k=k)
    gold = str(row_dict["answer"])

    return {
        "id": row_dict["id"],
        "choices_len": k,
        "answer": gold,
        "predicted_answer": predicted_answer,
        "is_correct": predicted_answer == gold,
        "generated_text": generated_text,

        "top5_candidates": top5_candidates,

        "digit_probs_1_to_k": digit_probs,  
        "digit_margin_top1_minus_top2": digit_margin,
        "digit_top1": digit_info["digit_top1"],
        "digit_top2": digit_info["digit_top2"],

        "prompt": prompt_text,
    }


def process_dataset(
    df: pd.DataFrame,
    builder: PromptBuilder,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: str,
    max_new_tokens: int,
    desc: str = "Processing",
) -> pd.DataFrame:
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        row_dict = row.to_dict()
        result = generate_for_row_with_top5(
            row_dict=row_dict,
            builder=builder,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DPO Builder: Generate inference results from SFT model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Override SFT adapter path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override training data path"
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=None,
        help="Override validation split ratio"
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        data_path=args.data_path,
        valid_ratio=args.valid_ratio,
    )