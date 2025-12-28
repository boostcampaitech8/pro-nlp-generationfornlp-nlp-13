from trl import SFTTrainer, SFTConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

class BaseSFTTrainer:
    """
    TRL 라이브러리의 SFTTrainer를 래핑하여 모델 타입별 맞춤형 학습을 수행하는 클래스.
    """
    
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, data_collator, 
                 peft_config, compute_metrics, preprocess_logits_for_metrics, 
                 training_args: dict, model_type: str):
        """
        Trainer를 초기화하고 학습에 필요한 설정 및 데이터셋을 구성합니다.
        
        Args:
            model: 학습할 사전 학습된 모델 또는 PEFT 모델
            tokenizer: 텍스트 처리에 사용할 토크나이저
            train_dataset: 학습용 데이터셋
            eval_dataset: 검증용 데이터셋
            data_collator: 배치 생성을 위한 데이터 콜레이터
            peft_config: LoRA 등 PEFT 설정 객체
            compute_metrics: 평가 지표 계산 함수
            preprocess_logits_for_metrics: 로짓 전처리 함수
            training_args: 전체 학습 설정이 담긴 딕셔너리 (train_cfg)
            model_type: 학습할 모델의 유형 ("knowledge" 또는 "inferential")
        """
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
        """
        공통 설정과 모델별 특화 설정을 조합하여 SFTConfig 객체를 생성합니다.
        
        Args:
            training_arg: 전체 학습 설정 딕셔너리
            model_type: 설정을 추출할 모델의 키값
            
        Returns:
            학습 인자가 적용된 SFTConfig 인스턴스
        """
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
            "fp16": training_arg["common"]["fp16"],                  # V100 필수
            "bf16": training_arg["common"]["bf16"],
            "optim": training_arg["common"]["optim"],  #  32bit 사용 (안정성) / paged_adamw_8bit -> 더 안된다면 이걸로!
            "weight_decay": training_arg["common"]["weight_decay"],
            "logging_steps": training_arg["common"]["logging_steps"],
            "save_strategy": training_arg["common"]["save_strategy"],
            "save_steps": training_arg["common"]["save_steps"],
            "eval_strategy": training_arg["common"]["eval_strategy"],
            "eval_steps": training_arg["common"]["eval_steps"],
            "save_total_limit": training_arg["common"]["save_total_limit"],
            "save_only_model": training_arg["common"]["save_only_model"],
            "report_to": training_arg["common"]["report_to"],
            "metric_for_best_model": training_arg["common"]["metric_for_best_model"],
            # "greater_is_better": training_arg["common"]["greater_is_better"],
        }

        return SFTConfig(**default_config)
    
    def train(self):
        """
        설정된 환경을 바탕으로 모델 학습을 실행합니다.
        
        Returns:
            학습 결과 정보가 담긴 TrainOutput 객체
        """
        return self.trainer.train()