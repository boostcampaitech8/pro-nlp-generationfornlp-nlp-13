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
from src.training.metrics import get_preprocess_logits_for_metrics, get_compute_metrics
from src.training.base_trainer import BaseSFTTrainer

def load_configs():
    """
    데이터, 모델, 훈련 설정이 포함된 YAML 파일들을 로드합니다.
    
    Returns:
        data_cfg, model_cfg, train_cfg 설정을 담은 튜플
    """
    with open("config/data_config.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("config/model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    with open("config/training_config.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg, train_cfg

def main():
    """
    지식형(4지선다) 모델을 위한 데이터 로딩, 토크나이징 및 SFT 학습 전과정을 실행합니다.
    """
    # 1. 설정 로드
    data_cfg, model_cfg, train_cfg = load_configs()
    
    # 2. 데이터 로딩 및 분류 (4지선다 추출)
    loader = DataLoader(data_cfg['data']['train_csv'])
    df = loader.load_and_flatten()
    
    classifier = QuestionClassifier()
    knowledge_df, _ = classifier.split_by_type(df)
    
    print(f"4지선다(지식형) 데이터: {len(knowledge_df)}개")
    
    # 3. 데이터 전처리 (Prompt -> Tokenize -> Split)
    prompt_formatter = PromptFormatter()
    processor = DatasetProcessor(prompt_formatter)
    knowledge_dataset = processor.process(knowledge_df)
    
    k_model_cfg = model_cfg['model']['knowledge']
    tokenizer_wrapper = TokenizerWrapper(
        model_name=k_model_cfg['model_name']
    )
    
    tokenized_dataset = tokenizer_wrapper.tokenize_dataset(knowledge_dataset)
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= data_cfg['data']['max_token_length'])  
    
    split_data = tokenized_dataset.train_test_split(test_size=data_cfg['data']['test_size'], seed=42)
    train_dataset, eval_dataset = split_data["train"], split_data["test"]
    
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # 4. 모델 및 LoRA 설정 로딩
    model = ModelLoader.load_model(
        model_name=k_model_cfg['model_name'],
        torch_dtype=k_model_cfg['torch_dtype'],
        device_map=k_model_cfg['device_map']
    )
    
    lora_params = model_cfg['lora']['knowledge']
    peft_config = LoraConfigFactory.create_default_config(**lora_params)
    
    # 5. Data Collator (Instruction 부분 마스킹)
    data_collator = CollatorFactory.create_completion_only_collator(
        tokenizer_wrapper.tokenizer
    )
    
    # 6. Trainer 생성 및 학습 시작
    trainer = BaseSFTTrainer(
        model=model,
        tokenizer=tokenizer_wrapper.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        peft_config=peft_config,
        compute_metrics=get_compute_metrics(tokenizer_wrapper.tokenizer),
        preprocess_logits_for_metrics=get_preprocess_logits_for_metrics(tokenizer_wrapper.tokenizer),
        training_args=train_cfg["training"],
        model_type="knowledge",
    )
    
    trainer.train()

if __name__ == "__main__":
    main()