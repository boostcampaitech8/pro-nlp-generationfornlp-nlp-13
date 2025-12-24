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
from src.data.classifier import ChoiceClassifier

# --- Config 로더 ---
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# --- Torch Dtype 매핑 ---
def get_torch_dtype(dtype_str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    else:
        return torch.float32

# --- 프롬프트 템플릿 (변경 없음) ---
PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요."""

PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요."""

def main():
    # 1. Config 로드
    config_path = os.path.join("config", "inference_config.yaml")
    cfg = load_config(config_path)
    
    print(f">>> Configuration Loaded: {cfg['project']['name']}")
    
    # 하드웨어 설정
    device_map = cfg['hardware']['device_map']
    torch_dtype = get_torch_dtype(cfg['hardware']['torch_dtype'])
    
    # 2. Dual Model 초기화
    print(">>> Loading Dual Models...")
    dual_model = DualModel(
        inferential_ckpt=cfg['models']['inferential']['checkpoint_path'],
        knowledge_ckpt=cfg['models']['knowledge']['checkpoint_path'],
        device_map=device_map
        # 참고: torch_dtype 전달이 필요하다면 DualModel 및 하위 클래스 __init__ 수정 필요
        # 현재는 기본적으로 모델 내부에서 처리하거나 config를 전달하도록 수정 가능
    )

    # 3. 데이터 로드 및 전처리
    test_data_path = cfg['paths']['test_data']
    print(f">>> Loading Test Data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)
    
    dataset_records = []
    classifier = ChoiceClassifier()
    
    for _, row in test_df.iterrows():
        problems = literal_eval(row['problems'])
        choices = problems['choices']
        
        # Classifier를 이용해 선지 개수 파악 (4지 vs 5지)
        len_choices = classifier.get_choice_count(choices)
        
        choices_string = "\n".join([f"{idx + 1} - {c}" for idx, c in enumerate(choices)])
        
        question_plus = problems.get('question_plus', None)
        if question_plus:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row['paragraph'],
                question=problems['question'],
                question_plus=question_plus,
                choices=choices_string,
            )
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row['paragraph'],
                question=problems['question'],
                choices=choices_string,
            )

        dataset_records.append({
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ],
            "len_choices": len_choices
        })

    # 4. 추론 실행
    print(">>> Starting Inference...")
    infer_results = []
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    
    for data in tqdm(dataset_records):
        _id = data["id"]
        messages = data["messages"]
        len_choices = data["len_choices"]
        
        # DualModel 라우팅 및 추론
        logits, tokenizer = dual_model.predict(messages, len_choices)
        
        # Logits -> Probability 변환
        target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(target_logit_list, dtype=torch.float32), 
                dim=0
            )
            .detach()
            .cpu()
            .numpy()
        )
        
        predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
        infer_results.append({"id": _id, "answer": predict_value})

    # 5. 결과 저장
    output_path = cfg['paths']['output']
    print(f">>> Saving results to {output_path}")
    pd.DataFrame(infer_results).to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()