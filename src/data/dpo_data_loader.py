import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from datasets import Dataset


@dataclass(frozen=True)
class DPODataConfig:
    train_path: str
    eval_path: Optional[str] = None
    seed: int = 42
    csv_dir: Optional[str] = None
    margin_threshold: float = 0.995
    eval_ratio: float = 0.1


def load_dpo_dataset(
    dpo_cfg: DPODataConfig,
) -> Dict[str, Dataset]:

    if not Path(dpo_cfg.train_path).exists():
        print("=" * 80)
        print("JSONL files not found. Building from CSV...")
        print("=" * 80)
        _build_from_csv(dpo_cfg)

    print(f"Loading DPO train dataset from: {dpo_cfg.train_path}")
    train_data = _load_jsonl(dpo_cfg.train_path)

    datasets = {
        "train": Dataset.from_list(train_data)
    }

    if dpo_cfg.eval_path:
        print(f"Loading DPO eval dataset from: {dpo_cfg.eval_path}")
        eval_data = _load_jsonl(dpo_cfg.eval_path)
        datasets["validation"] = Dataset.from_list(eval_data)

    return datasets


def _build_from_csv(dpo_cfg: DPODataConfig):
    """CSV에서 JSONL 자동 생성"""
    from src.data.dpo_dataset import build_dpo_dataset, save_jsonl
    import pandas as pd
    from sklearn.model_selection import train_test_split

    csv_dir = Path(dpo_cfg.csv_dir) if dpo_cfg.csv_dir else Path("./data/dpo_outputs")
    train_gen_path = csv_dir / "train_gen.csv"
    valid_gen_path = csv_dir / "valid_gen.csv"

    if not train_gen_path.exists() or not valid_gen_path.exists():
        raise FileNotFoundError(
            f"CSV files not found at {csv_dir}\n"
            f"Please run dpo_builder.py first:\n"
            f"  python -m src.data.dpo_builder --config config.yaml"
        )

    print(f"Loading CSV from: {csv_dir}")
    train_gen_df = pd.read_csv(train_gen_path)
    valid_gen_df = pd.read_csv(valid_gen_path)
    print(f"Train Gen: {len(train_gen_df)} rows")
    print(f"Valid Gen: {len(valid_gen_df)} rows\n")

    print(f"Building DPO pairs (margin_threshold={dpo_cfg.margin_threshold})...")
    all_pairs = build_dpo_dataset(
        train_gen_df=train_gen_df,
        valid_gen_df=valid_gen_df,
        margin_threshold=dpo_cfg.margin_threshold,
    )

    print(f"\nSplitting into train/eval (ratio={dpo_cfg.eval_ratio})...")
    train_pairs, eval_pairs = train_test_split(
        all_pairs,
        test_size=dpo_cfg.eval_ratio,
        random_state=dpo_cfg.seed,
    )
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Eval pairs: {len(eval_pairs)}\n")

    print("Saving JSONL files...")
    save_jsonl(train_pairs, dpo_cfg.train_path)
    if dpo_cfg.eval_path:
        save_jsonl(eval_pairs, dpo_cfg.eval_path)

    print("=" * 80)
    print("DPO dataset built successfully!")
    print("=" * 80 + "\n")


def _load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
