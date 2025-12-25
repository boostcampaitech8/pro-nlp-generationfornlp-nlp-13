import sys
import os
import yaml
import pandas as pd
import numpy as np
import torch
from ast import literal_eval
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.models.dual_model import DualModel
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
    config 디렉토리의 YAML 파일들을 로드하여 설정 정보들을 반환합니다.
    
    Returns:
        data_cfg, model_cfg, inference_cfg 설정을 담은 튜플
    """
    with open("config/data_config.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("config/model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    with open("config/inference_config.yaml", "r") as f:
        inference_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg, inference_cfg


def main():
    """
    전체 추론 파이프라인(설정 로드, 모델 초기화, 데이터 처리, 추론, 저장)을 실행합니다.
    """
    # 1. Config 로드
    data_cfg, model_cfg, inference_cfg = load_configs()
    
    # 2. Dual Model 초기화
    print(">>> Loading Dual Models...")
    dual_model = DualModel(
        inferential_ckpt=inference_cfg['models']['inferential']['checkpoint_path'],
        knowledge_ckpt=inference_cfg['models']['knowledge']['checkpoint_path'],
        device_map='auto'
    )

    # 3. 데이터 로드 및 전처리
    loader = DataLoader(data_cfg['data']['test_csv'])
    test_df = loader.load_and_flatten()

    print(f"테스트 데이터: {len(test_df)}개")

    # 프롬프트 생성 및 Dataset 변환
    prompt_formatter = PromptFormatter()
    processor = DatasetProcessor(prompt_formatter)
    test_dataset = processor.process(test_df, is_test=True)

    # 4. 추론 실행
    print(">>> 추론 시작!")
    infer_results = []
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    
    for data in tqdm(test_dataset):
        _id = data["id"]
        messages = data["messages"]
        len_choices = data["len_choices"]
        
        # DualModel 라우팅 및 추론
        probs = dual_model.predict(messages, len_choices)
        
        predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
        infer_results.append({"id": _id, "answer": predict_value})

    # 5. 결과 저장
    output_path = inference_cfg['paths']['output']
    pd.DataFrame(infer_results).to_csv(output_path, index=False)
    print(f">>> 결과 저장 완료! : {output_path}")


if __name__ == "__main__":
    main()