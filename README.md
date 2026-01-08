# 수능형 문제 풀이 모델 생성

## 1. 프로젝트 개요
### 1.1 프로젝트 주제
 본 프로젝트는 수능형 국어·사회 지문 기반 객관식 문제를 해결하는 대회 과제로, 입력(지문/문항/선택지)에 대해 정답 선택지 번호를 예측하는 모델을 구축하였다. GPT·Claude 등 대형 모델 대비 상대적으로 작은 규모의 모델로도 경쟁력 있는 성능을 달성하는 것을 목표로 하였다.

### 1.2 주요 전략 및 방법론
EDA를 통해 4/5지선다 문항이 지문 길이와 정답 분포 등에서 서로 다른 특성을 갖는 것을 기반으로 유형 분리 전략을 적용하였다. 이후 외부 데이터 증강, 프롬프트 최적화, Qwen3-14B와 A.X-4.0 Light의 soft voting 앙상블, 그리고 마진 기반 저신뢰도 문항에 대한 Critic 재추론을 결합하여 최종 추론 파이프라인을 구성하였다.

### 1.3 데이터셋
수능 국어 및 사회 탐구 영역의 복합적인 추론 능력을 평가하기 위한 데이터셋 활용


* **수능형 문항**: 수능의 국어, 사회 영역(윤리, 정치, 사회)과 비슷한 문제
* **KMMLU** (Korean History), **MMMLU** (HighSchool 데이터 중 역사, 경제, 정치, 지리, 심리)
* **KLUE MRC**(경제, 교육산업, 국제, 부동산, 사회, 생활, 책마을)



## 2. 팀원 소개 및 역할
| 이름 | 프로필 | 역할 |
| :---: | :---: | --- |
| **김영현**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/Kimyoung-hyun) | <img src="./assets/kim.jpg" width="100"> | 데이터 증강, 프롬프트 엔지니어링 |
| **윤준상**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/JunandSang) | <img src="./assets/yun.jpg" width="100"> | 데이터 분석, 데이터 증강, 프롬프트 엔지니어링 |
| **장세현**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/sucruba70) | <img src="./assets/jang.jpg" width="100"> | 데이터 분석, 전체 파이프라인 설계 및 구현, DPO 실험, 추론 로직 구현 |
| **주현민**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/zoosumzoosum) | <img src="./assets/zoo.jpg" width="100"> | 데이터 증강, 프롬프트 엔지니어링, EDA |
| **한지석**<br>[<img src="./assets/github-mark.png" width="25">](https://github.com/jis-archive) | <img src="./assets/han.jpg" width="100"> | 디렉토리 구조 설계, 앙상블 구현, 추론 성능 실험 |

## 3. 결과

### 최종 리더보드 (Private)

<img width="1000" alt="image" src="./assets/private.png">

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
세팅을 직접하고 싶다면 `config.yaml` 를 참고해주세요.

**train**


```bash
python -m src.train.py
```

**inference**


```python
python -m src.inference
```

## 7. Wrap-Up Report
프로젝트 전반의 시행착오와 솔루션 및 회고는 [MRC_NLP_13.pdf](MRC_NLP_13.pdf)을 통해 확인할 수 있습니다.
