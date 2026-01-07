# 수능형 문제 풀이 모델 생성

## 1. 프로젝트 개요
### 1.1 프로젝트 주제

본 프로젝트는 거대 언어 모델(LLM)에 비해 상대적으로 파라미터 수가 적은 소형 모델이 한국어 특유의 복잡한 맥락과 수능 시험의 고차원적 추론 논리를 얼마나 정교하게 수행할 수 있는지 탐색합니다. 범용 대형 모델의 한국어 최적화 한계를 극복하기 위해, 한국어 도메인 지식과 수능 문항의 특성을 반영한 특화 모델을 구축하여 대형 모델에 준하는 추론 효율성을 달성하는 것을 목표로 합니다.

### 1.2 데이터셋
수능 국어 및 사회 탐구 영역의 복합적인 추론 능력을 평가하기 위한 데이터셋 활용


* **수능형 문항**: 수능의 국어, 사회 영역(윤리, 정치, 사회)과 비슷한 문제
* **KMMLU** (Korean History), **MMMLU** (HighSchool 데이터 중 역사, 경제, 정치, 지리, 심리)
* **KLUE MRC**(경제, 교육산업, 국제, 부동산, 사회, 생활, 책마을)



## 2. 팀원 소개 및 역할
| 이름 | 프로필 | 역할 |
| :---: | :---: | --- |
| **김영현**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/Kimyoung-hyun) | <img src="./assets/kim.jpg" width="100"> | 내용입력 |
| **윤준상**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/JunandSang) | <img src="./assets/yun.jpg" width="100"> | 내용입력 |
| **장세현**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/sucruba70) | <img src="./assets/jang.jpg" width="100"> | 내용입력 |
| **주현민**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/zoosumzoosum) | <img src="./assets/zoo.jpg" width="100"> | 내용입력 |
| **한지석**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/jis-archive) | <img src="./assets/han.jpg" width="100"> | 내용입력 |

## 3. 결과

### 최종 리더보드 (Private)

<img width="1216" alt="image" src="./assets/private.png">

## 4. 파이프라인
사진 추가

## 5. 디렉토리 구조

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

## 6. Train 및 Inference 실행

**train**

세팅을 직접하고 싶다면 `config.yaml` 를 참고해주세요.

모델 학습

```bash
python -m src.train.py
```


**inference**

모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다.

- 학습한 모델의 test_dataset에 대한 결과를 제출하기 위해선 추론(`-do_predict`)만 진행하면 됩니다.
- 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(`-do_eval`)를 진행하면 됩니다.

```python
python -m src.inference
```

## 7. Wrap-Up Report
프로젝트 전반의 시행착오와 솔루션 및 회고는 [MRC_NLP_13.pdf](MRC_NLP_13.pdf)을 통해 확인할 수 있습니다.
