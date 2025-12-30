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
from src.models.knowledge_model import KnowledgeModel
from src.models.inferential_model import InferentialModel
from src.utils.seed import set_seed



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

DIGIT_IDS = [16, 17, 18, 19, 20]  # '1'~'5'
THINK_END_ID = 151668  # </think>

def get_answer_from_logits(outputs, input_len):
    """
    output_scores에서 </think> 이후 첫 digit의 logits 확인
    """
    if not hasattr(outputs, 'scores') or not outputs.scores:
        return None
    
    generated_ids = outputs.sequences[0]  # (total_len,)
    
    # </think> 위치 찾기
    think_end_positions = (generated_ids == THINK_END_ID).nonzero(as_tuple=True)[0]
    
    if len(think_end_positions) == 0:
        # </think>가 없으면 생성된 토큰 중 첫 digit 찾기
        for i in range(input_len, len(generated_ids)):
            token_id = generated_ids[i].item()
            if token_id in DIGIT_IDS:
                step_idx = i - input_len
                if 0 <= step_idx < len(outputs.scores):
                    step_logits = outputs.scores[step_idx][0]  # (V,)
                    digit_logits = step_logits[DIGIT_IDS]  # (5,)
                    return digit_logits.argmax().item() + 1  # 1~5
        return None
    
    # </think> 이후 첫 digit 토큰 찾기
    think_end_pos = think_end_positions[-1].item()
    
    for i in range(think_end_pos + 1, len(generated_ids)):
        token_id = generated_ids[i].item()
        if token_id in DIGIT_IDS:
            # 이 토큰이 생성된 step의 logits 확인
            step_idx = i - input_len
            if 0 <= step_idx < len(outputs.scores):
                step_logits = outputs.scores[step_idx][0]  # (V,)
                digit_logits = step_logits[DIGIT_IDS]  # (5,)
                # 가장 높은 확률의 digit 선택
                predicted = digit_logits.argmax().item() + 1  # 1~5
                return predicted
    
    return None


def parse_pred_fallback(text):
    """
    logits에서 실패 시 텍스트 파싱 (fallback)
    """
    if "</think>" in text:
        after_think = text.split("</think>")[-1]
        for char in after_think:
            if char in ['1', '2', '3', '4', '5']:
                return char
    
    for keyword in ["정답:", "정답은", "Answer:"]:
        if keyword in text:
            after_keyword = text.split(keyword)[-1]
            for char in after_keyword:
                if char in ['1', '2', '3', '4', '5']:
                    return char
    
    for char in reversed(text):
        if char in ['1', '2', '3', '4', '5']:
            return char
    
    return '1'

def to_test_text( example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=True, 
    )
    return {"text": text}

def main():
    """
    전체 추론 파이프라인(설정 로드, 모델 초기화, 데이터 처리, 추론, 저장)을 실행합니다.
    """
    # 1. Config 로드, seed 설정, wandb 초기화
    data_cfg, model_cfg, inference_cfg = load_configs()

    set_seed(42)
    
    # 2. Dual Model 초기화
    print(">>> Loading Dual Models...")
    # dual_model = DualModel(
    #     inferential_ckpt=inference_cfg['models']['inferential']['checkpoint_path'],
    #     knowledge_ckpt=inference_cfg['models']['knowledge']['checkpoint_path'],
    #     device_map='auto'
    # )
    k_model = KnowledgeModel(inference_cfg['models']['knowledge']['checkpoint_path'])
    i_model = InferentialModel(inference_cfg['models']['inferential']['checkpoint_path'])
    tokenizer = k_model.get_tokenizer()

    # 3. 데이터 로드 및 전처리
    loader = DataLoader(data_cfg['data']['test_csv'])
    test_df = loader.load_and_flatten()
    classifier = QuestionClassifier()
    test_df = classifier.classify_dataset(test_df)

    print(f"테스트 데이터: {len(test_df)}개")

    # 프롬프트 생성 및 Dataset 변환
    prompt_formatter = PromptFormatter()
    processor = DatasetProcessor(prompt_formatter)
    test_dataset = processor.process(test_df, is_test=True)

    test_ds_text = test_dataset.map(
        to_test_text,
        fn_kwargs={"tokenizer": tokenizer} # 여기에 전달할 인자를 딕셔너리로 넣음
    )
    print(test_ds_text)
    # 4. 추론 실행
    print(">>> 추론 시작!")
    infer_results = []
    logits_success = 0
    fallback_used = 0

    print("🚀 Logits 기반 추론 시작...")

    k_model = k_model.get_model()
    i_model = i_model.get_model()
    k_model.eval()
    i_model.eval()
    with torch.inference_mode():
        for ex in tqdm(test_ds_text):
            _id = ex["id"]
            text = ex["text"]
            

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to("cuda")

            if ex['question_type'] == 'knowledge':
                outputs = k_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,       
                )
            else:
                outputs = i_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,       
                )

            input_len = inputs["input_ids"].shape[-1]
            
            # 1차 시도: logits 기반
            pred = get_answer_from_logits(outputs, input_len)
            
            if pred is not None:
                logits_success += 1
                pred_str = str(pred)
            else:
                # 2차 시도: 텍스트 파싱
                fallback_used += 1
                gen_ids = outputs.sequences[0][input_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                pred_str = parse_pred_fallback(gen_text)
            
            infer_results.append({
                "id": _id,
                "answer": pred_str
            })

    # 5. 결과 저장
    output_path = inference_cfg['paths']['output']
    pd.DataFrame(infer_results).to_csv(output_path, index=False)
    print(f">>> 결과 저장 완료! : {output_path}")

if __name__ == "__main__":
    main()