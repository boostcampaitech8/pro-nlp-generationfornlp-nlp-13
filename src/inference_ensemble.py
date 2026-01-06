import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import re
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
    
    model_cfg, prompt_cfg, inference_cfg = create_configs(cfg_dict)
    
    adapter_path = adapter_path or inference_cfg["adapter_path"]
    output_path = output_path or inference_cfg["output_path"]
    output_logits_path = inference_cfg["output_logits_path"]
    test_data_path = inference_cfg["test_data_path"]
    max_new_tokens = inference_cfg.get("max_new_tokens", 100)
    
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
    
    builder = PromptBuilder(prompt_cfg)
    print("PromptBuilder ready!\n")
    
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)
    results = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(test_df)}]")
        
        row_dict = row.to_dict()
        result = process_row(
            row_dict,
            builder,
            tokenizer,
            model,
            device=device,
            max_new_tokens=max_new_tokens
        )
        results.append(result)
    
    print(f"\n" + "=" * 60)
    print(f"Saving predictions to {output_path}...")
    full_predictions_df = pd.DataFrame(results)
    predictions_df = full_predictions_df[["id", "answer"]]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    print(f"Saved {len(predictions_df)} predictions")
    print("=" * 60)


    logits_df = full_predictions_df[["id", "digit_logits", "digit_probs"]]

    Path(output_logits_path).parent.mkdir(parents=True, exist_ok=True)
    logits_df.to_csv(output_logits_path, index=False)

    print(f"Saved {len(logits_df)} predictions")
    print("=" * 60)
    
    return predictions_df, logits_df


def extract_answer(text: str) -> str:
    numbers = re.findall(r'[1-5]', text)
    
    if numbers:
        return numbers[-1]
        
    return "no"

def get_logits_and_prob(step_logits: torch.Tensor, tokenizer, k: int) -> Dict[str, Any]:
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

    return {
        "digit_logits": digit_logits.tolist(),
        "digit_probs": digit_probs.tolist(),
    }

def process_row(
    row_dict: Dict,
    builder: PromptBuilder,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: str = "cuda",
    max_new_tokens: int = 100,
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

    k = int(row_dict["choices_len"])
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True, 
        )
    
    full_text = tokenizer.decode(output_ids.sequences[0][input_len:], skip_special_tokens=True)

    step_logits = output_ids.scores[-2][0]
    
    digit_info = get_logits_and_prob(step_logits, tokenizer, k)

    answer = extract_answer(full_text)
    
    return {
        "id": row_dict.get("id"),
        "answer": answer,
        "full_output": full_text,
        "digit_logits": digit_info["digit_logits"],
        "digit_probs": digit_info["digit_probs"]
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
    
    inference_cfg = cfg_dict.get("inference", {})
    
    return model_cfg, prompt_cfg, inference_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using config.yaml"
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
