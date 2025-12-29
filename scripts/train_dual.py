import yaml
import gc
import torch
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
from src.utils.seed import set_seed
from src.utils.wandb import wandb_init, wandb_finish

def load_configs():
    """
    학습에 필요한 데이터, 모델, 훈련 설정 YAML 파일들을 로드합니다.
    
    Returns:
        data_cfg, model_cfg, train_cfg 설정을 담은 튜플
    """
    with open("config/data_config.yaml", "r") as f: data_cfg = yaml.safe_load(f)
    with open("config/model_config.yaml", "r") as f: model_cfg = yaml.safe_load(f)
    with open("config/training_config.yaml", "r") as f: train_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg, train_cfg

def train_model(model_type, dataset_df, data_cfg, model_cfg, train_cfg):
    """
    지정된 모델 타입(knowledge/inferential)에 맞춰 전처리, 모델 로드, 학습 및 메모리 정리를 수행합니다.
    
    Args:
        model_type: 학습할 모델의 유형 ("knowledge" 또는 "inferential")
        dataset_df: 해당 유형으로 분류된 데이터가 담긴 DataFrame
        data_cfg: 데이터 관련 설정 딕셔너리
        model_cfg: 모델 및 LoRA 관련 설정 딕셔너리
        train_cfg: Trainer 및 TrainingArgs 관련 설정 딕셔너리
    """
    print(f"\n>>> {model_type.upper()} 모델 학습 시작")

    # 0. wandb 초기화
    # >>> [TODO] 원하는 설정값으로 수정하세요!!
    wandb_config = {
        # 모델 설정 (model_cfg에서 추출)
        f"{model_type}_model": model_cfg['model'][model_type]['model_name'],
        
        # LoRA 하이퍼파라미터 (lora_params에서 추출)
        "lora_r": model_cfg['lora'][model_type]['r'],
        "lora_alpha": model_cfg['lora'][model_type]['lora_alpha'],
        
        # 학습 설정 (train_cfg에서 추출)
        f"{model_type}_lr": train_cfg['training'][model_type].get('learning_rate'),
        f"{model_type}_batch_size": train_cfg['training'][model_type].get('per_device_train_batch_size'),
        f"{model_type}_epochs": train_cfg['training'][model_type].get('num_train_epochs'),

        # 기타
        "seed": 42
    }
    
    run_name = ""   # >>> [TODO] 추가하세요!
    wandb_init(config=wandb_config, project_name="My_LLM_Project", run_name=run_name)
    
    # 1. 데이터 전처리 (Prompt -> Tokenize -> Split)
    prompt_formatter = PromptFormatter()
    processor = DatasetProcessor(prompt_formatter)
    dataset = processor.process(dataset_df)
    
    m_cfg = model_cfg['model'][model_type]
    tokenizer_wrapper = TokenizerWrapper(
        model_name=m_cfg['model_name']
    )
    
    tokenized_dataset = tokenizer_wrapper.tokenize_dataset(dataset)
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= data_cfg['data']['max_token_length'])
    split_data = tokenized_dataset.train_test_split(test_size=data_cfg['data']['test_size'], seed=42)
    train_dataset, eval_dataset = split_data["train"], split_data["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # 2. 모델 및 설정 로드
    model = ModelLoader.load_model(
        model_name=m_cfg['model_name'],
        torch_dtype=m_cfg['torch_dtype'],
        device_map=m_cfg['device_map'],
        is_quantization=m_cfg['is_quantization']
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # 3. LoRA Config (model_config.yaml의 lora 세션 기반)
    lora_params = model_cfg['lora'][model_type]
    peft_config = LoraConfigFactory.create_default_config(**lora_params)

    # 4. Data Collator (Instruction 부분 마스킹)
    data_collator = CollatorFactory.create_completion_only_collator(tokenizer_wrapper.tokenizer)

    # 5. Trainer 생성 및 학습 시작
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
        model_type=model_type,
    )
    
    trainer.train()
    wandb_finish()

    # 6. 메모리 정리 (중요: 다음 모델 학습을 위해 GPU 메모리 비우기)
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    """
    데이터를 로드하고 문제 유형별로 분류한 뒤, 지식형과 추론형 모델을 순차적으로 학습시킵니다.
    """
    data_cfg, model_cfg, train_cfg = load_configs()

    set_seed(42)
    
    # 1. 데이터 로드 및 분류
    loader = DataLoader(data_cfg['data']['train_csv'])
    df = loader.load_and_flatten()
    classifier = QuestionClassifier()
    knowledge_df, inferential_df = classifier.split_by_type(df)

    # 2. 지식형(4지선다) 학습
    print(f"4지선다(지식형) 데이터: {len(knowledge_df)}개")
    train_model("knowledge", knowledge_df, data_cfg, model_cfg, train_cfg)
    
    # 3. 추론형(5지선다) 학습
    print(f"5지선다(추론형) 데이터: {len(inferential_df)}개")
    train_model("inferential", inferential_df, data_cfg, model_cfg, train_cfg)

if __name__ == "__main__":
    main()