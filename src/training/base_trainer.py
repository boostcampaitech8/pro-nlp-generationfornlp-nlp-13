from trl import SFTTrainer, SFTConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

class BaseSFTTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        data_collator,
        peft_config,
        compute_metrics,
        preprocess_logits_for_metrics,
        training_args: dict,  # train_cfg 딕셔너리를 통째로 받음
        model_type: str       # "knowledge" 또는 "inferential" 지정
    ):
        self.sft_config = self._create_sft_config(training_args, model_type)
        
        self.trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            args=self.sft_config,
        )
    
    def _create_sft_config(self, training_arg, model_type) -> SFTConfig:
        """SFTConfig 생성"""

        default_config = {
            "do_train": training_arg["common"]["do_train"],
            "do_eval": training_arg["common"]["do_eval"],
            "lr_scheduler_type": training_arg["common"]["lr_scheduler_type"],
            "max_seq_length": training_arg["common"]["max_seq_length"],
            "output_dir": training_arg[model_type]["output_dir"],
            "per_device_train_batch_size": training_arg[model_type]["per_device_train_batch_size"],
            "per_device_eval_batch_size": training_arg[model_type]["per_device_eval_batch_size"],
            "num_train_epochs": training_arg[model_type]["num_train_epochs"],
            "learning_rate": float(training_arg[model_type]["learning_rate"]),
            "weight_decay": training_arg["common"]["weight_decay"],
            "logging_steps": training_arg["common"]["logging_steps"],
            "save_strategy": training_arg["common"]["save_strategy"],
            "eval_strategy": training_arg["common"]["eval_strategy"],
            "save_total_limit": training_arg["common"]["save_total_limit"],
            "save_only_model": training_arg["common"]["save_only_model"],
            "report_to": training_arg["common"]["report_to"],
        }

        return SFTConfig(**default_config)
    
    def train(self):
        """학습 실행"""
        return self.trainer.train()