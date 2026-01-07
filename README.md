# README

# 수능형 문제 풀이 모델 생성



## Wrap-Up Report

[MRC_NLP_13.pdf](MRC_NLP_13.pdf)

프로젝트 전반의 시행착오와 솔루션 및 회고는 렙업 리포트를 통해 확인할 수 있습니다.

# 1. Project Overview

| 항목 | 내용 |
| --- | --- |
| 프로젝트 주제 | 내용입력 |
| 프로젝트 구성 | 내용입력 |
| 평가 지표 | 내용입력 |
| 진행 기간 | 2025년 12월 15일 ~ 2026년 1월 6일 |

## 팀원

| 김영현 | 윤준상 | 장세현 | 주현민 | 한지석 |
| --- | --- | --- | --- | --- |
| <img src="./assets/kim.jpg" width="100"> | <img src="./assets/yun.jpg" width="100"> | <img src="./assets/jang.jpg" width="100"> | <img src="./assets/zoo.jpg" width="100"> | <img src="./assets/han.jpg" width="100"> |
| [Kimyoung-hyun](https://github.com/Kimyoung-hyun) | [JunandSang](https://github.com/JunandSang) | [sucruba70](https://github.com/sucruba70) | [zoosumzoosum](https://github.com/zoosumzoosum) | [jis-archive](https://github.com/jis-archive) |

## 역할

| 이름 | 역할 |
| --- | --- |
| 김영현 | 내용입력 |
| 윤준상 | 내용입력 |
| 장세현 | 내용입력 |
| 주현민 | 내용입력 |
| 한지석 | 내용입력 |

# 2. Result

### 최종 리더보드 (Private)

<img width="1216" alt="image" src="./assets/private.png">

# 3. **Architecture**

```bash
.
src/
├─ train.py
├─ inference.py
│
├─ training/
│   ├─ trainer.py
│   └─ model_loader.py
│
├─ data/
│   ├─ preprocessor.py
│   ├─ data_loader.py
│   └─ tokenizer_wrapper.py
│
├─ prompt/
│   ├─ prompt_builder.py
│   ├─ prompt_registry.py 
│   └─ templates
│      ├─ system
│      └─ user
│
├─ utils/
│   ├─ metrics.py
│ 	├─ wandb.py
│   └─ seed.py
│    
└─ notebooks/
```

# 훈련 평가 추론

**train**

만약 arguments 에 대한 세팅을 직접하고 싶다면 `arguments.py` 를 참고해주세요.

retriever 구축

```bash
python -m src.retriever.retriever_pipeline
```

reader 학습

```bash
python -m src.mrc.run_mrc_training
```

**inference**

retrieval 과 mrc 모델의 학습이 완료되면 `inference_with_reranker.py` 를 이용해 odqa 를 진행할 수 있습니다.

- 학습한 모델의 test_dataset에 대한 결과를 제출하기 위해선 추론(`-do_predict`)만 진행하면 됩니다.
- 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(`-do_eval`)를 진행하면 됩니다.

```python
python -m src.inference_with_reranker
```
