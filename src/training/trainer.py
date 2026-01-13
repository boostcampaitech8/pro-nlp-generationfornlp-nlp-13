from dataclasses import dataclass
from typing import Optional, Literal

from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from src.utils.metrics import Metrics


@dataclass(frozen=True)
class TrainerConfig:
    output_dir: str
    max_seq_length: int = 2048
    packing: bool = False

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

@dataclass(frozen=True)
class TokenFormatConfig:  # 이름 변경으로 기존 PromptConfig와 충돌 방지
    response_template: str = "<|im_start|>assistant\n"
    label_pos_from_tail: int = 3
    logit_pos_from_tail: int = 4

def build_trainer(
    trainer_cfg: TrainerConfig,
    token_cfg: TokenFormatConfig,
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
) -> SFTTrainer:
    
    training_args = TrainingArguments(
        output_dir=trainer_cfg.output_dir,
        num_train_epochs=trainer_cfg.num_train_epochs,
        per_device_train_batch_size=trainer_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=trainer_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,

        learning_rate=trainer_cfg.learning_rate,
        optim=trainer_cfg.optim,
        lr_scheduler_type=trainer_cfg.lr_scheduler_type,
        weight_decay=trainer_cfg.weight_decay,
        warmup_ratio=trainer_cfg.warmup_ratio,
        max_grad_norm=trainer_cfg.max_grad_norm,

        fp16=trainer_cfg.fp16,
        bf16=trainer_cfg.bf16,

        logging_steps=trainer_cfg.logging_steps,
        eval_steps=trainer_cfg.eval_steps,
        save_steps=trainer_cfg.save_steps,
        eval_strategy=trainer_cfg.eval_strategy,
        save_strategy=trainer_cfg.save_strategy,
        save_total_limit=trainer_cfg.save_total_limit,
        load_best_model_at_end=trainer_cfg.load_best_model_at_end,
        metric_for_best_model=trainer_cfg.metric_for_best_model,
        greater_is_better=trainer_cfg.greater_is_better,

        report_to=trainer_cfg.report_to,
        seed=trainer_cfg.seed,
        remove_unused_columns=trainer_cfg.remove_unused_columns,
    )

    response_template = token_cfg.response_template
    print(f"response_template : {response_template}")

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    metrics = Metrics(
        tokenizer=tokenizer, 
        label_pos_from_tail=token_cfg.label_pos_from_tail, 
        logit_pos_from_tail=token_cfg.logit_pos_from_tail
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=trainer_cfg.max_seq_length,
        packing=trainer_cfg.packing,
        compute_metrics=metrics.compute_metrics,
        preprocess_logits_for_metrics=metrics.preprocess_logits_for_metrics,
        dataset_text_field="text",
    )
    return trainer
