"""
4지선다(지식형) 모델 학습 스크립트 (YAML 설정 연동 버전)
"""
import yaml
from src.data.data_loader import DataLoader 
from src.data.classifier import QuestionClassifier
from src.data.prompt_formatter import PromptFormatter
from src.data.dataset_processor import DatasetProcessor
from src.data.tokenizer_wrapper import TokenizerWrapper
from src.data.collator import CollatorFactory
from src.models.model_loader import ModelLoader
from src.models.lora_config import LoraConfigFactory
from src.training.metrics import preprocess_logits_for_metrics, compute_metrics
from src.training.base_trainer import BaseSFTTrainer

def load_configs():
    """YAML 파일들을 로드하여 딕셔너리로 반환"""
    with open("config/data_config.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("config/model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    with open("config/training_config.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg, train_cfg

def main():
    # 0. 설정 로드
    data_cfg, model_cfg, train_cfg = load_configs()
    
    # 1. 데이터 로딩 및 분류
    # data_config.yaml의 train_csv 경로 사용
    loader = DataLoader(data_cfg['data']['train_csv'])
    df = loader.load_and_flatten()
    
    classifier = QuestionClassifier()
    _, inferential_df = classifier.split_by_type(df)
    
    print(f"5지선다(지식형) 데이터: {len(inferential_df)}개")
    
    # 2. 프롬프트 생성 및 Dataset 변환
    prompt_formatter = PromptFormatter()
    processor = DatasetProcessor(prompt_formatter)
    knowledge_dataset = processor.process(inferential_df)
    
    # 3. 토크나이저 로드 (data_config.yaml 기반)
    tokenizer_info = data_cfg['data']['tokenizer']
    tokenizer_wrapper = TokenizerWrapper(
        model_name=tokenizer_info['model_name'],
        padding_side=tokenizer_info['padding_side']
    )
    
    # 4. 필터링 및 분할 (data_config.yaml 기반)
    tokenized_dataset = tokenizer_wrapper.tokenize_dataset(knowledge_dataset)
    
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= data_cfg['data']['max_token_length'])  
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=data_cfg['test_size'], seed=42)

    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['test']
    
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # 5. 모델 로딩 (model_config.yaml 기반)
    k_model_cfg = model_cfg['model']['knowledge_model']
    model = ModelLoader.load_model(
        model_name=k_model_cfg['base_model'],
        torch_dtype=k_model_cfg['torch_dtype'],
        device_map=k_model_cfg['device_map']
    )
    
    # 6. LoRA Config (model_config.yaml의 lora.knowledge 세션 기반)
    lora_params = model_cfg['lora']['knowledge']
    peft_config = LoraConfigFactory.create_config(lora_params)
    
    # 7. Data Collator
    data_collator = CollatorFactory.create_completion_only_collator(
        tokenizer_wrapper.tokenizer
    )
    
    # 9. Trainer 생성 및 학습 (training_config.yaml의 common + knowledge 조합)
    common_cfg = train_cfg['training']['common']
    inferential_cfg = train_cfg['training']['inferential']

    # sft_config = SFTConfig(
    #     do_train=True,
    #     do_eval=True,
    #     lr_scheduler_type=common_cfg['lr_scheduler_type'],
    #     max_seq_length=1024,
    #     output_dir=knowledge_cfg['output_dir'],
    #     per_device_train_batch_size=knowledge_cfg['per_device_train_batch_size'],
    #     per_device_eval_batch_size=1,
    #     num_train_epochs=knowledge_cfg['num_train_epochs'],
    #     learning_rate=float(knowledge_cfg['learning_rate']),
    #     weight_decay=0.01,
    #     logging_steps=common_cfg['logging_steps'],
    #     save_strategy=common_cfg['save_strategy'],
    #     eval_strategy=common_cfg['eval_strategy']
    #     save_total_limit=2,
    #     save_only_model=True,
    #     report_to="none",
    # )

    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     data_collator=data_collator,
    #     tokenizer=TokenizerWrapper.tokenizer,
    #     compute_metrics=compute_metrics,
    #     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    #     peft_config=peft_config,
    #     args=sft_config,
    # )

    trainer = BaseSFTTrainer(
        model=model,
        tokenizer=tokenizer_wrapper.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        peft_config=peft_config,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        output_dir=inferential_cfg["output_dir"],
        num_train_epochs=3,
        learning_rate=float(inferential_cfg['learning_rate']),
    )

    # 4. Trainer 생성 (매우 깔끔해집니다!)
    trainer = BaseSFTTrainer(
        model=model,
        tokenizer=tokenizer_wrapper.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        peft_config=peft_config,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        training_args=train_cfg["training"],
        model_type="inferential",
    )
    
    trainer.train()

if __name__ == "__main__":
    main()