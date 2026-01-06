from dataclasses import dataclass
from typing import Optional, Literal

from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from src.utils.metrics import compute_metrics, preprocess_logits_for_metrics


@dataclass(frozen=True)
class TrainerConfig:
    output_dir: str
    max_seq_length: int = 2048
    packing: bool =False

    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    learning_rate: float = 2e-4
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    fp16: bool = True
    bf16: bool = False

    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    logging_steps: int = 20
    eval_steps: int = 20
    save_steps: int = 20

    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True

    report_to: Optional[str] = None
    seed: int = 42
    remove_unused_columns: bool = False

def build_trainer(
    cfg: TrainerConfig,
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
) -> SFTTrainer:
    
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        learning_rate=cfg.learning_rate,
        optim=cfg.optim,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,

        fp16=cfg.fp16,
        bf16=cfg.bf16,

        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,

        report_to=cfg.report_to,
        seed=cfg.seed,
        remove_unused_columns=cfg.remove_unused_columns,
    )

    response_template = "<|im_start|><|assistant|>"

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        dataset_text_field="text",
    )
    return trainer