
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
import re
import numpy as np
from tqdm import tqdm 

import pandas as pd
import torch
from transformers import AutoTokenizer

from src.data.preprocessor import parse_problems_column, add_choices_len
from src.prompt.prompt_builder import PromptBuilder, PromptConfig
from src.training.model_loader import ModelConfig, load_model_inference


def main(
    config_path: str,
    adapter_path: str = None,
    output_path: str = None,
):

    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    model_cfg, prompt_cfg, inference_cfg, prompt_cfg_retry = create_configs(cfg_dict)
    
    adapter_path = adapter_path or inference_cfg["adapter_path"]
    output_path = output_path or inference_cfg["output_path"]
    output_logits_path = inference_cfg["output_logits_path"]
    test_data_path = inference_cfg["test_data_path"]
    max_new_tokens = inference_cfg.get("max_new_tokens", 30)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    print(f"Loading test data from {test_data_path}...")
    test_df = load_test_data(test_data_path)
    print(f"Loaded {len(test_df)} rows\n")
    
    print(f"Loading tokenizer from {model_cfg.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {adapter_path}...")
    model = load_model_inference(model_cfg, adapter_path)
    model.eval()
    print("Model loaded successfully!\n")
    
    builder_1 = PromptBuilder(prompt_cfg)
    builder_2 = PromptBuilder(prompt_cfg_retry) 
    print("PromptBuilder ready!\n")
    
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)

    pass1 = []
    logit_gaps = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Pass1"):
        row_dict = row.to_dict()

        r1 = process_row(
            row_dict,
            builder_1,
            tokenizer,
            model,
            device=device,
            max_new_tokens=max_new_tokens
        )
        pass1.append((row_dict, r1))
        logit_gaps.append(r1["logit_gap"])

    quantile = inference_cfg.get("retry_quantile", 0.25)
    thr = compute_logit_gap_threshold(logit_gaps, quantile=quantile)
    print(f"[2-step] quantile={quantile} -> logit_gap_threshold={thr:.4f}")

    final_results = []
    retry_cnt = 0

    for row_dict, r1 in tqdm(pass1, desc="Pass2"):
        # 초기값은 r1의 결과로 설정
        final_answer = r1["answer"]
        final_probs = r1["digit_probs"]
        final_gap = r1["logit_gap"]
        retry_type = "r1"
        
        need_retry = (r1["answer"] == "no") or (r1["logit_gap"] <= thr)

        if need_retry:
            retry_cnt += 1
            retry_input = dict(row_dict)
            retry_input["pred_answer"] = final_answer 
            
            r2 = process_row_retry(
                retry_input,
                builder_2,
                tokenizer,
                model,
                device=device,
                max_new_tokens=max_new_tokens
            )
            
            # r2가 유효한 답변을 내놓은 경우 (r2 결과로 업데이트)
            if r2["answer_2"] != "no":
                final_answer = r2["answer_2"]
                # r2에서도 digit_probs와 logit_gap을 반환하도록 process_row_retry가 구성되어 있어야 합니다.
                final_probs = r2["digit_probs"]
                final_gap = r2.get("logit_gap", r1["logit_gap"])
                retry_type = "r2"

            else:
                # r2도 "no"라면 r1의 확률 기반으로 강제 선택 (probs/gap은 r1 유지)
                k = int(row_dict.get("choices_len", 5))
                final_answer = str(int(np.argmax(r1["digit_probs"][:k])) + 1)

        final_results.append({
            "id": row_dict.get("id"),
            "answer": final_answer,
            "digit_probs": final_probs, # r2가 성공했다면 r2의 확률값
            "logit_gap": final_gap,      # r2가 성공했다면 r2의 갭
            "r": retry_type, 
        })

    print(f"[2-step] retried {retry_cnt}/{len(test_df)} rows ({retry_cnt/len(test_df)*100:.1f}%)")

    predictions_df = pd.DataFrame(final_results)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # full_output_path = output_path.replace('.csv', '_full.csv')
    # predictions_df_full = predictions_df[["id", "answer", "digit_probs", "logit_gap"]]
    # predictions_df_full.to_csv(full_output_path, index=False)
    # print(f"Saved full results to {full_output_path}")

    predictions_df_submit = predictions_df[["id", "answer"]]
    predictions_df_submit.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")
    print("=" * 60)

    logits_df = predictions_df[["id", "digit_probs", "r"]]

    Path(output_logits_path).parent.mkdir(parents=True, exist_ok=True)
    logits_df.to_csv(output_logits_path, index=False)

    print(f"Saved {len(logits_df)} predictions")
    print("=" * 60)
    
    return predictions_df


def extract_answer(text: str, k: int = 5) -> str:
    numbers = re.findall(rf"[1-{k}]", text)
    return numbers[-1] if numbers else "no"


def compute_logit_gap_threshold(
    logit_gaps: List[float],
    quantile: float = 0.25
) -> float:
    gaps = [g for g in logit_gaps if g is not None and not np.isnan(g)]
    if not gaps:
        raise ValueError("Empty logit_gap list")
    return float(np.quantile(gaps, quantile))


def process_row(
    row_dict: Dict,
    builder: PromptBuilder,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: str = "cuda",
    max_new_tokens: int = 30,
) -> Dict:

    output = builder.build_message(row_dict)
    messages = output["messages"]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)

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
    full_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    step_logits = outputs.scores[-2][0]

    k = int(row_dict.get("choices_len", 5))

    topk_values, topk_indices = torch.topk(step_logits, k=k)
    probs_full = torch.softmax(step_logits, dim=-1)

    topk_candidates = []
    for rank, (logit_val, token_id) in enumerate(zip(topk_values, topk_indices)):
        topk_candidates.append({
            "rank": rank + 1,
            "token_id": token_id.item(),
            "token": tokenizer.decode([token_id.item()]),
            "logit": logit_val.item(),
            "prob": probs_full[token_id].item(),
        })

    digit_tokens = [str(i) for i in range(1, k + 1)]
    digit_token_ids = []

    for digit in digit_tokens:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        digit_token_ids.append(encoded[0])

    digit_logits = torch.tensor([step_logits[tid].item() for tid in digit_token_ids])
    digit_probs = torch.softmax(digit_logits, dim=-1)

    top2_digit_values, top2_digit_indices = torch.topk(digit_logits, k=min(2, k))
    if k >= 2:
        logit_gap = (top2_digit_values[0] - top2_digit_values[1]).item()
    else:
        logit_gap = 0.0

    answer = extract_answer(generated_text, k=k)

    return {
        "id": row_dict.get("id"),
        "answer": answer,
        "digit_probs": digit_probs.tolist(),
        "logit_gap": logit_gap,
        "generated_text": generated_text,
        "prompt": prompt_text
    }


def process_row_retry(
    row_dict: Dict,
    builder: PromptBuilder,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: str = "cuda",
    max_new_tokens: int = 30,
) -> Dict:
    
    output = builder.build_message(row_dict)
    messages = output["messages"]
    k = int(row_dict.get("choices_len", 5))

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)

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

    full_text = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True)
    ans2 = extract_answer(full_text, k=k)

    step_logits = outputs.scores[-2][0]

    digit_tokens = [str(i) for i in range(1, k + 1)]
    digit_token_ids = []

    for digit in digit_tokens:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        digit_token_ids.append(encoded[0])

    digit_logits = torch.tensor([step_logits[tid].item() for tid in digit_token_ids])
    digit_probs = torch.softmax(digit_logits, dim=-1)


    return {
        "id": row_dict.get("id"),
        "answer_2": ans2,
        "full_text_2": full_text,
        "digit_probs": digit_probs.tolist()
    }


def load_test_data(test_path: Path) -> pd.DataFrame:
    """Load and preprocess test data."""
    test_df = pd.read_csv(test_path)
    test_df = parse_problems_column(test_df)
    test_df = add_choices_len(test_df)
    return test_df


def create_configs(cfg_dict: Dict[str, Any]) -> tuple:
    model_cfg_dict = cfg_dict["model"].copy()
    model_cfg_dict["use_gradient_checkpointing"] = False
    model_cfg = ModelConfig(**model_cfg_dict)

    prompt_dict = cfg_dict["inference"]["prompt"]
    prompt_cfg = PromptConfig(
        policy=prompt_dict["policy"],
        mode="test",
        verbose=False
    )

    retry_prompt_dict = cfg_dict["inference"]["retry_prompt"]
    prompt_cfg_retry = PromptConfig(
        policy=retry_prompt_dict["policy"],
        mode="test",
        verbose=False
    )

    inference_cfg = cfg_dict.get("inference", {})

    return model_cfg, prompt_cfg, inference_cfg, prompt_cfg_retry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using config.yaml"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_ax.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Override adapter path (e.g., ./outputs/reading/final_model)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path (e.g., predictions.csv)"
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        adapter_path=args.adapter_path,
        output_path=args.output,
    )
