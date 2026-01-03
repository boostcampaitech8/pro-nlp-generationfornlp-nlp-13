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


def load_dpo_dataset(
    dpo_cfg: DPODataConfig,
) -> Dict[str, Dataset]:

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


def _load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
