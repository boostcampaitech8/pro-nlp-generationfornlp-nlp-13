import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DPODatasetConfig:
    train_gen_path: str
    valid_gen_path: str
    output_train_path: str
    output_eval_path: str
    margin_threshold: float = 0.995
    eval_ratio: float = 0.1
    seed: int = 42


def main(cfg: DPODatasetConfig):
    print("=" * 80)
    print("DPO Dataset Builder - Create chosen/rejected pairs")
    print("=" * 80)
    print(f"Train Gen CSV: {cfg.train_gen_path}")
    print(f"Valid Gen CSV: {cfg.valid_gen_path}")
    print(f"Margin Threshold: {cfg.margin_threshold}")
    print(f"Eval Ratio: {cfg.eval_ratio}")
    print(f"Seed: {cfg.seed}")
    print("=" * 80 + "\n")

    print("Loading CSV files...")
    train_gen_df = pd.read_csv(cfg.train_gen_path)
    valid_gen_df = pd.read_csv(cfg.valid_gen_path)
    print(f"Train Gen: {len(train_gen_df)} rows")
    print(f"Valid Gen: {len(valid_gen_df)} rows\n")

    print("Building DPO pairs...")
    all_pairs = build_dpo_dataset(
        train_gen_df=train_gen_df,
        valid_gen_df=valid_gen_df,
        margin_threshold=cfg.margin_threshold,
    )

    print(f"\nSplitting into train/eval (ratio={cfg.eval_ratio})...")
    train_pairs, eval_pairs = train_test_split(
        all_pairs,
        test_size=cfg.eval_ratio,
        random_state=cfg.seed,
    )
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Eval pairs: {len(eval_pairs)}\n")

    print("Saving JSONL files...")
    save_jsonl(train_pairs, cfg.output_train_path)
    save_jsonl(eval_pairs, cfg.output_eval_path)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


def wrap_answer(digit: str) -> str:
    return f"<think>\n\n</think>\n\n{digit}"


def normalize_digit(token_text: str, k: int) -> Optional[str]:
    if token_text is None:
        return None
    m = re.findall(rf"[1-{k}]", str(token_text))
    return m[-1] if m else None


def pick_rejected_digits(row: Dict[str, Any], num_rej: int) -> List[str]:
    k = int(row["choices_len"])
    gold = str(row["answer"])

    picked: List[str] = []

    top5_candidates = row.get("top5_candidates")
    if isinstance(top5_candidates, str):
        try:
            top5_candidates = ast.literal_eval(top5_candidates)
        except:
            top5_candidates = []

    for c in (top5_candidates or []):
        d = normalize_digit(c.get("token"), k)
        if d is None or d == gold or d in picked:
            continue
        picked.append(d)
        if len(picked) >= num_rej:
            return picked

    for d in map(str, range(1, k + 1)):
        if d == gold or d in picked:
            continue
        picked.append(d)
        if len(picked) >= num_rej:
            break

    return picked


def make_pairs_for_row(row: Dict[str, Any], num_rej: int, source: str) -> List[Dict[str, Any]]:
    prompt = row["prompt"]
    gold = str(row["answer"])
    k = int(row["choices_len"])

    rejected_digits = pick_rejected_digits(row, num_rej=num_rej)

    pairs = []
    for rd in rejected_digits:
        pairs.append({
            "prompt": prompt,
            "chosen": wrap_answer(gold),
            "rejected": wrap_answer(rd),
            "meta": {
                "id": row.get("id"),
                "choices_len": k,
                "source": source,
                "margin": row.get("digit_margin_top1_minus_top2"),
                "gold": gold,
                "rejected_digit": rd,
                "is_correct": bool(row.get("is_correct")),
            }
        })
    return pairs


def build_dpo_dataset(
    train_gen_df: pd.DataFrame,
    valid_gen_df: pd.DataFrame,
    margin_threshold: float = 0.995,
) -> List[Dict[str, Any]]:

    all_df = pd.concat([train_gen_df, valid_gen_df], ignore_index=True)

    incorrect_df = all_df[all_df["is_correct"] == False]

    soft_true_df = all_df[
        (all_df["is_correct"] == True) &
        (all_df["digit_margin_top1_minus_top2"] <= margin_threshold)
    ]

    print(f"Incorrect samples: {len(incorrect_df)}")
    print(f"Soft True samples (margin <= {margin_threshold}): {len(soft_true_df)}")

    all_pairs: List[Dict[str, Any]] = []

    for _, r in incorrect_df.iterrows():
        row = r.to_dict()
        k = int(row["choices_len"])
        num_rej = 2 if k == 4 else 3
        all_pairs.extend(make_pairs_for_row(row, num_rej=num_rej, source="incorrect"))

    for _, r in soft_true_df.iterrows():
        row = r.to_dict()
        all_pairs.extend(make_pairs_for_row(row, num_rej=1, source="soft_true"))

    print(f"Total DPO pairs generated: {len(all_pairs)}")

    if len(all_pairs) == 0:
        raise ValueError(
            "No DPO training pairs generated!\n"
            f"  - Incorrect samples: {len(incorrect_df)}\n"
            f"  - Soft True samples (margin <= {margin_threshold}): {len(soft_true_df)}\n"
            "Try lowering margin_threshold or check SFT model accuracy."
        )

    return all_pairs


def save_jsonl(data: List[Dict[str, Any]], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved: {output_path} ({len(data)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build DPO dataset from dpo_builder.py outputs"
    )
    parser.add_argument(
        "--train-gen",
        type=str,
        default="./data/dpo_outputs/train_gen.csv",
        help="Path to train_gen.csv from dpo_builder.py"
    )
    parser.add_argument(
        "--valid-gen",
        type=str,
        default="./data/dpo_outputs/valid_gen.csv",
        help="Path to valid_gen.csv from dpo_builder.py"
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default="./data/dpo_train.jsonl",
        help="Output path for train JSONL"
    )
    parser.add_argument(
        "--output-eval",
        type=str,
        default="./data/dpo_eval.jsonl",
        help="Output path for eval JSONL"
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.995,
        help="Margin threshold for soft_true samples (default: 0.995)"
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Ratio of eval split (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/eval split"
    )

    args = parser.parse_args()

    cfg = DPODatasetConfig(
        train_gen_path=args.train_gen,
        valid_gen_path=args.valid_gen,
        output_train_path=args.output_train,
        output_eval_path=args.output_eval,
        margin_threshold=args.margin_threshold,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    main(cfg)
