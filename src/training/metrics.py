import numpy as np
import torch
import evaluate
import re

def get_preprocess_logits_for_metrics(tokenizer):
    """
    평가 과정에서 메모리 효율을 위해 로짓 중 정답 후보 토큰에 해당하는 값만 추출하는 함수를 생성합니다.
    
    Args:
        tokenizer: 정답 후보("1"~"5")의 보카 인덱스를 찾기 위한 토크나이저
        
    Returns:
        필요한 로짓만 필터링하여 반환하는 preprocess_logits_for_metrics 함수
    """
    def preprocess_logits_for_metrics(logits, labels):
        """
        Args:
            logits: 모델의 전체 출력 로짓
            labels: 정답 레이블
            
        Returns:
            마지막 정답 예측 시점(-2 토큰)의 후보군("1"~"5") 로짓값
        """
        logits = logits if not isinstance(logits, tuple) else logits[0]
        # 후보 토큰들("1", "2", "3", "4", "5")의 인덱스 리스트
        logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"], tokenizer.vocab["3"], tokenizer.vocab["4"], tokenizer.vocab["5"]]
        
        # -2: answer token 시점, -1: eos token 시점 (모델 출력 구조에 따름)
        logits = logits[:, -2, logit_idx]
        return logits
    
    return preprocess_logits_for_metrics


def get_compute_metrics(tokenizer):
    """
    디코딩된 레이블과 예측된 로짓을 비교하여 Macro F1-score를 계산하는 함수를 생성합니다.
    
    Args:
        tokenizer: 레이블 디코딩을 위한 토크나이저
        
    Returns:
        F1-score를 산출하여 반환하는 compute_metrics 함수
    """
    # metric 로드
    f1_metric = evaluate.load("f1")

    # 텍스트 형태의 정답을 숫자 인덱스로 변환하는 맵
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    def compute_metrics(evaluation_result):
        """
        Args:
            evaluation_result: (필터링된 logits, labels) 튜플
            
        Returns:
            {'f1': 스코어} 형태의 평가 결과 딕셔너리
        """
        logits, labels = evaluation_result

        # 1. 레이블 데이터 전처리: padding 제거 및 텍스트 디코딩
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Qwen3 등의 thinking 모델 대응: <think> 태그 제거 후 실제 정답 텍스트만 추출
        labels = list(map(lambda x: x.split("</think>")[-1].strip(), labels))
        labels = list(map(lambda x: int_output_map[x], labels))

        # 2. 로짓 데이터를 확률 분포로 변환 및 최대값 선택
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # 3. Macro F1-score 계산
        f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

        return f1
    
    return compute_metrics
