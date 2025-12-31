import numpy as np
import torch


DIGIT_IDS = [16, 17, 18, 19, 20]


def compute_metrics(eval_pred, label_pos_from_tail: int = 3):
    """
    eval_pred:
      - (predictions, label_ids) or EvalPrediction
      - predictions: (B, 5)  # preprocess_logits_for_metrics에서 이미 '1'~'5' logits만 뽑아둔 상태
      - label_ids: (B, L)    # -100은 ignore index

    label_pos_from_tail:
      - "정답 토큰('1'~'5')"이 labels의 끝에서 몇 번째인지(1-based)
      - 예) labels tail이 [..., '정답', <|im_end|>] 라면 '정답'은 끝에서 2번째 -> label_pos_from_tail=2
      - 예) labels tail이 [..., '정답', <|im_end|>, '\n'] 라면 '정답'은 끝에서 3번째 -> label_pos_from_tail=3
    """
    if hasattr(eval_pred, "predictions"):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        preds, labels = eval_pred

    preds_t = torch.as_tensor(preds)  # (B, 5)
    pred_cls = torch.argmax(preds_t, dim=-1).cpu().numpy().astype(np.int64)  # 0~4

    labels_t = torch.as_tensor(labels)  # (B, L)
    not_ignored = labels_t.ne(-100)     # (B, L)

    # 각 샘플의 "진짜 길이"(= 마지막 labels!=-100 위치 + 1)
    rev = torch.flip(not_ignored, dims=[1])
    last_true_from_end = torch.argmax(rev.int(), dim=1)  # (B,)
    has_any = not_ignored.any(dim=1)                     # (B,)
    L = labels_t.size(1)
    real_len = L - last_true_from_end
    real_len = torch.where(has_any, real_len, torch.full_like(real_len, L))

    # 정답 토큰 위치 = real_len - label_pos_from_tail
    pos = (real_len - label_pos_from_tail).clamp(min=0, max=L - 1)  # (B,)
    batch_idx = torch.arange(labels_t.size(0), device=labels_t.device)
    gold_tok = labels_t[batch_idx, pos].cpu().numpy().astype(np.int64)

    # token id -> class(0~4): DIGIT_IDS[0]이 '1'이라 가정
    gold_cls = gold_tok - DIGIT_IDS[0]

    valid = (gold_cls >= 0) & (gold_cls < 5)
    pred_cls = pred_cls[valid]
    gold_cls = gold_cls[valid]

    acc = float((pred_cls == gold_cls).mean()) if len(gold_cls) > 0 else 0.0

    # macro-f1
    f1s = []
    for c in range(5):
        tp = np.sum((pred_cls == c) & (gold_cls == c))
        fp = np.sum((pred_cls == c) & (gold_cls != c))
        fn = np.sum((pred_cls != c) & (gold_cls == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1}

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