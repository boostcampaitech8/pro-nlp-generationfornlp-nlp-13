import numpy as np
import torch
import evaluate
import re

DIGIT_IDS = [16, 17, 18, 19, 20]  # '1'~'5'

def get_preprocess_logits_for_metrics(tokenizer):
    """
    평가 과정에서 메모리 효율을 위해 로짓 중 정답 후보 토큰에 해당하는 값만 추출하는 함수를 생성합니다.
    
    Args:
        tokenizer: 정답 후보("1"~"5")의 보카 인덱스를 찾기 위한 토크나이저
        
    Returns:
        필요한 로짓만 필터링하여 반환하는 preprocess_logits_for_metrics 함수
    """
    def preprocess_logits_for_metrics(logits, labels, pos_from_tail=4):
        """
        반환: (batch, 5)  -> '1'~'5'에 해당하는 logits만 뽑아서 metrics 단계로 전달
        """
        # Trainer가 (logits, ...) 튜플을 줄 때가 있어서 정리
        if isinstance(logits, tuple):
            logits = logits[0]  # (B, L, V)

        # labels: (B, L), pad/무시 영역은 -100일 가능성이 큼
        # real_len = 마지막으로 labels != -100 인 위치 + 1 로 복원
        labels_t = torch.as_tensor(labels)
        not_ignored = (labels_t != -100)

        # 샘플별로 마지막 not_ignored 위치 찾기
        # (뒤에서부터 True 찾기)
        rev = torch.flip(not_ignored, dims=[1])
        last_true_from_end = torch.argmax(rev.int(), dim=1)          # (B,)
        has_any = not_ignored.any(dim=1)                             # (B,)
        # real_len = seq_len - last_true_from_end
        seq_len = labels_t.size(1)
        real_len = seq_len - last_true_from_end

        # 만약 labels가 전부 -100인 샘플이 있으면(비정상) 그냥 seq_len로 처리
        real_len = torch.where(has_any, real_len, torch.full_like(real_len, seq_len))

        pos = (real_len - pos_from_tail).clamp(min=0, max=seq_len-1) # (B,)

        # (B, V)로 해당 위치의 logits만 gather
        logits_t = torch.as_tensor(logits)                           # (B, L, V)
        batch_idx = torch.arange(logits_t.size(0), device=logits_t.device)
        picked = logits_t[batch_idx, pos, :]                         # (B, V)

        # digit ids만 슬라이스 -> (B, 5)
        picked_digits = picked[:, DIGIT_IDS]
        return picked_digits
    
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

    def compute_metrics(eval_pred, label_pos_from_tail=3):
        """
        eval_pred:
        - (predictions, label_ids) 튜플 형태가 가장 흔함
        - predictions: preprocess_logits_for_metrics가 반환한 (B, 5)
        - label_ids: (B, L) with -100 ignored
        반환: {"accuracy": ..., "macro_f1": ...}
        """
        if hasattr(eval_pred, "predictions"):
            preds, labels = eval_pred.predictions, eval_pred.label_ids
        else:
            preds, labels = eval_pred

        preds_t = torch.as_tensor(preds)
        pred_cls = torch.argmax(preds_t, dim=-1).cpu().numpy().astype(np.int64)  # (B,)

        labels_t = torch.as_tensor(labels)

        not_ignored = (labels_t != -100)
        rev = torch.flip(not_ignored, dims=[1])
        last_true_from_end = torch.argmax(rev.int(), dim=1)
        has_any = not_ignored.any(dim=1)

        seq_len = labels_t.size(1)
        real_len = seq_len - last_true_from_end
        real_len = torch.where(has_any, real_len, torch.full_like(real_len, seq_len))

        pos_label = (real_len - label_pos_from_tail).clamp(min=0, max=seq_len - 1)
        batch_idx = torch.arange(labels_t.size(0), device=labels_t.device)
        gold_tok = labels_t[batch_idx, pos_label].cpu().numpy().astype(np.int64) 

        gold_cls = gold_tok - DIGIT_IDS[0]  

        valid = (gold_cls >= 0) & (gold_cls < 5)
        pred_cls = pred_cls[valid]
        gold_cls = gold_cls[valid]

        acc = (pred_cls == gold_cls).mean() if len(gold_cls) > 0 else 0.0

        f1s = []
        for c in range(5):
            tp = np.sum((pred_cls == c) & (gold_cls == c))
            fp = np.sum((pred_cls == c) & (gold_cls != c))
            fn = np.sum((pred_cls != c) & (gold_cls == c))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1s.append(f1)

        macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0

        return {"accuracy": float(acc), "f1": macro_f1}
        
    return compute_metrics
