import numpy as np
import torch


DIGIT_IDS = [57, 58, 59, 60, 61]

def preprocess_logits_for_metrics(logits, labels):
    """
    정답 시작 위치(labels가 -100이 아닌 첫 지점)의 '1'~'5' logits만 추출
    """
    if isinstance(logits, tuple):
        logits = logits[0]

    labels_t = torch.as_tensor(labels)

    not_ignored = (labels_t != -100)
    first_token_indices = not_ignored.int().argmax(dim=1) # (batch_size,)

    batch_idx = torch.arange(logits.size(0), device=logits.device)
    picked_logits = logits[batch_idx, first_token_indices, :] # (batch_size, vocab_size)

    # '1'~'5'에 해당하는 logits만 슬라이싱 (B, 5)
    return picked_logits[:, DIGIT_IDS]

def compute_metrics(eval_pred):
    """
    '1'~'5' 정답 토큰에 대한 Accuracy 및 Macro-F1 계산
    """
    if hasattr(eval_pred, "predictions"):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        preds, labels = eval_pred

    # 1예측값 결정 (0~4 index)
    preds_t = torch.as_tensor(preds)
    pred_cls = torch.argmax(preds_t, dim=-1).cpu().numpy()

    # 2. 실제 정답(Gold) 추출
    labels_t = torch.as_tensor(labels)
    not_ignored = (labels_t != -100)
    first_token_indices = not_ignored.int().argmax(dim=1)
    
    batch_idx = torch.arange(labels_t.size(0), device=labels_t.device)
    gold_tok = labels_t[batch_idx, first_token_indices].cpu().numpy().astype(np.int64)

    # token id -> class(0~4) 변환
    gold_cls = gold_tok - DIGIT_IDS[0]

    # 3. 유효한 범위(1~5) 내의 데이터만 필터링 (에러 방지)
    valid = (gold_cls >= 0) & (gold_cls < 5)
    pred_cls = pred_cls[valid]
    gold_cls = gold_cls[valid]

    acc = float((pred_cls == gold_cls).mean()) if len(gold_cls) > 0 else 0.0

    f1s = []
    for c in range(5):
        tp = np.sum((pred_cls == c) & (gold_cls == c))
        fp = np.sum((pred_cls == c) & (gold_cls != c))
        fn = np.sum((pred_cls != c) & (gold_cls == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0
    }