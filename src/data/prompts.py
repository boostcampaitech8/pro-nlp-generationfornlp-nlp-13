# src/reading/prompts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class PromptBundle:
    system: str
    user_no_plus: str
    user_plus: str

PROMPTS: Dict[str, PromptBundle] = {}

# ---------------------------
# basic (현재 기본)
# ---------------------------
PROMPTS["basic"] = PromptBundle(
    system="""당신은 '언어 이해' 및 '비문학 독해' 영역의 전문가입니다.
주어진 [지문]의 내용을 절대적인 사실로 간주하고, 이를 바탕으로 [질문]에 대한 가장 적절한 답을 논리적으로 도출하십시오.
외부 지식이나 주관적인 추측은 철저히 배제하고, 오직 텍스트에 기반하여 정답을 선택하십시오.

[출력 형식]
- 출력은 정확히 한 줄이어야 합니다.
- 형식은 정확히 다음과 같아야 합니다: 정답: X
- X는 1~5 중 하나의 정수입니다.
- 위 한 줄 외에는 어떤 것도 출력하지 마십시오. (설명, 근거, 풀이 과정, 따옴표, 마침표, 공백/줄바꿈 추가 포함)
- 지문/선택지/질문을 절대 다시 출력하지 마십시오.
""",
    user_no_plus="""### 지문
{paragraph}

### 질문
{question}

### 선택지
{choices}

[출력]
정답: X
""",
    user_plus="""### 지문
{paragraph}

### 질문
{question}

### 선택지
{choices}

### 추가 자료
{question_plus}

[출력]
정답: X
""",
)

PROMPTS["cot"] = PromptBundle(
    system="""당신은 '언어 이해' 및 '비문학 독해' 영역의 최상위권 전문가입니다.
주어진 [지문]의 내용을 절대적인 사실로 간주하고, 이를 바탕으로 [질문]에 대한 가장 적절한 답을 논리적으로 도출하십시오.
외부 지식이나 주관적인 추측은 철저히 배제하고, 오직 텍스트에 기반하여 정답을 선택하십시오.

[문제 해결 절차]
1) 질문이 긍정형/부정형인지 먼저 판별하십시오.
2) 각 선택지의 핵심 단서를 지문에서 찾아 근거를 확보하십시오.
3) 지문과 선택지를 1:1로 대조하여 모순(Contradiction) 및 언급 없음(Not Mentioned)을 제거하십시오.
4) 남은 선택지 중 가장 확실한 근거를 가진 번호를 정답으로 선택하십시오.

[출력 규칙: 매우 중요]
- 최종 출력은 반드시 한 줄.
- 형식: 정답: X (X는 1~5)
- 풀이 과정/근거/설명/중간 판단은 절대 출력하지 마십시오. 최종 정답만 출력하십시오.
""",
    user_no_plus="""### 지문
{paragraph}

### 질문
{question}

### 선택지
{choices}

### 문제 해결 가이드라인
- 질문 유형(긍정/부정)을 먼저 확인하라.
- 선택지별 근거 문장을 지문에서 찾고, 지문과 대조해 오답을 제거하라.
- 지문에 없는 내용은 선택하지 마라.

[최종 출력]
정답: X
""",
    user_plus="""### 지문
{paragraph}

### 질문
{question}

### 선택지
{choices}

### 추가 자료
{question_plus}

### 문제 해결 가이드라인
- 질문 유형(긍정/부정)을 먼저 확인하라.
- 선택지별 근거 문장을 지문에서 찾고, 지문과 대조해 오답을 제거하라.
- 지문에 없는 내용은 선택하지 마라.

[최종 출력]
정답: X
""",
)

PROMPTS["selective_cot"] = PromptBundle(
    system="""당신은 대학수학능력시험 '언어 이해' 영역의 전문가입니다.
긴 지문을 읽고 질문에 답하십시오.

[사고 프로세스]
1. 질문의 핵심 키워드 파악
2. 지문에서 관련 부분만 빠르게 탐색
3. 각 선택지를 해당 부분과 대조
4. 명확히 틀린 선택지 빠르게 제거
5. 남은 선택지 중 정답 선택

[출력 형식]
정답: X

※ 설명이나 근거는 출력하지 마십시오.""",

    user_no_plus="""### 지문
{paragraph}

### 질문
{question}

### 선택지
{choices}

정답: """,

    user_plus="""### 지문
{paragraph}

### 보기
{question_plus}

### 질문
{question}

### 선택지
{choices}

정답: """
)

def get_prompt_bundle(version: str = "basic") -> PromptBundle:
    try:
        return PROMPTS[version]
    except KeyError as e:
        raise KeyError(
            f"Unknown prompt version: {version}. "
            f"Available: {sorted(PROMPTS.keys())}"
        ) from e