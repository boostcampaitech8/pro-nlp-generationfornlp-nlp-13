import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from peft import TaskType
from transformers import AutoTokenizer

from src.training.model_loader import ModelConfig, LoRAConfig, load_model
from src.training.trainer import TrainerConfig, build_trainer
from src.data.data_loader import DataConfig, make_train_valid_dataset
from src.prompt.prompt_builder import PromptConfig
from src.data.tokenizer_wrapper import TokenizerConfig
from src.utils.seed import set_seed
from src.utils.wandb import wandb_init, wandb_finish


def main(
    model_cfg: ModelConfig,
    lora_cfg: LoRAConfig,
    trainer_cfg: TrainerConfig,
    data_cfg: DataConfig,
    prompt_cfg: PromptConfig,
    tokenize_cfg_train: TokenizerConfig,
    tokenize_cfg_gen: TokenizerConfig,
    wandb_cfg: Dict[str, Any] = None,
):
    
    set_seed(trainer_cfg.seed)
    print(f"Random seed set to: {trainer_cfg.seed}")
    
    if wandb_cfg and wandb_cfg.get("enabled", False):
        wandb_init(
            config={
                "model": model_cfg.__dict__,
                "lora": lora_cfg.__dict__,
                "trainer": trainer_cfg.__dict__,
                "data": data_cfg.__dict__,
                "prompt": prompt_cfg.__dict__,
            },
            project_name=wandb_cfg.get("project_name", "ax-noname"),
            run_name=wandb_cfg.get("run_name") or f"run_{Path(trainer_cfg.output_dir).name}",
            entity=wandb_cfg.get("entity"),
        )
    
    print(f"Loading tokenizer from {model_cfg.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"pad_token not found, set to eos_token: {tokenizer.eos_token}")
    
    print("Preparing datasets...")
    datasets = make_train_valid_dataset(
        data_cfg=data_cfg,
        prompt_cfg=prompt_cfg,
        tokenize_cfg_train=tokenize_cfg_train,
        tokenize_cfg_gen=tokenize_cfg_gen,
        tokenizer=tokenizer,
    )
    
    print(f"Train dataset: {len(datasets['train'])}")
    if "validation" in datasets:
        print(f"Validation dataset: {len(datasets['validation'])}")
    
    print("Loading model...")
    model = load_model(model_cfg, lora_cfg)
    
    trainer = build_trainer(
        cfg=trainer_cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
    )
    
    print("=" * 60)
    print("Training Start")
    print("=" * 60)
    trainer.train()
    final_model_path = Path(trainer_cfg.output_dir) / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    if wandb_cfg and wandb_cfg.get("enabled", False):
        wandb_finish()

    print("-" * 60)
    print("Training Complete")
    print(f"Best model saved at: {trainer_cfg.output_dir}")
    print("-" * 60)


def create_configs(cfg_dict: Dict[str, Any]):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    model_dict = cfg_dict["model"].copy()
    if "compute_dtype" in model_dict and isinstance(model_dict["compute_dtype"], str):
        model_dict["compute_dtype"] = dtype_map.get(
            model_dict["compute_dtype"], 
            torch.float16
        )
    
    lora_dict = cfg_dict["lora"].copy()
    if "task_type" in lora_dict and isinstance(lora_dict["task_type"], str):
        lora_dict["task_type"] = TaskType.CAUSAL_LM
    
    model_cfg = ModelConfig(**model_dict)
    lora_cfg = LoRAConfig(**lora_dict)
    trainer_cfg = TrainerConfig(**cfg_dict["trainer"])
    data_cfg = DataConfig(**cfg_dict["data"])
    prompt_cfg = PromptConfig(**cfg_dict["prompt"])
    
    tokenize_cfg_train = TokenizerConfig(**cfg_dict["tokenizer"]["train"])
    tokenize_cfg_gen = TokenizerConfig(**cfg_dict["tokenizer"]["gen"])
    
    return (
        model_cfg,
        lora_cfg,
        trainer_cfg,
        data_cfg,
        prompt_cfg,
        tokenize_cfg_train,
        tokenize_cfg_gen,
    )


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A.X-8B with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="./config_ensemble.yaml",
        help="Path to config YAML file"
    )
    args = parser.parse_args()
    
    raw_cfg = load_config(args.config)
    (
        model_cfg,
        lora_cfg,
        trainer_cfg,
        data_cfg,
        prompt_cfg,
        tokenize_cfg_train,
        tokenize_cfg_gen,
    ) = create_configs(raw_cfg)
    
    main(
        model_cfg=model_cfg,
        lora_cfg=lora_cfg,
        trainer_cfg=trainer_cfg,
        data_cfg=data_cfg,
        prompt_cfg=prompt_cfg,
        tokenize_cfg_train=tokenize_cfg_train,
        tokenize_cfg_gen=tokenize_cfg_gen,
        wandb_cfg=raw_cfg.get("wandb"),
    )
