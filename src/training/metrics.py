import numpy as np
import torch
import evaluate

# preprocess_logits_for_metrics 함수를 반환하는 팩토리 함수
def get_preprocess_logits_for_metrics(tokenizer):
    """
    tokenizer를 클로저로 캡처하여 preprocess_logits_for_metrics 함수를 반환
    """
    def preprocess_logits_for_metrics(logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"], tokenizer.vocab["3"], tokenizer.vocab["4"], tokenizer.vocab["5"]]
        logits = logits[:, -2, logit_idx] # -2: answer token, -1: eos token
        return logits
    
    return preprocess_logits_for_metrics


# metric 계산 함수를 반환하는 팩토리 함수
def get_compute_metrics(tokenizer):
    """
    tokenizer를 클로저로 캡처하여 compute_metrics 함수를 반환
    """
    # metric 로드 (F1-score로 변경)
    f1_metric = evaluate.load("f1")
    
    # 정답 토큰 매핑
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    
    def compute_metrics(evaluation_result):
        logits, labels = evaluation_result

        # 토큰화된 레이블 디코딩
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
        labels = list(map(lambda x: int_output_map[x], labels))

        # 소프트맥스 함수를 사용하여 로그트 변환
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # F1-score 계산 (average='macro'로 설정하여 macro F1-score 계산)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
        
        return f1
    
    return compute_metrics
