from typing import Dict, List

class PromptFormatter:
    """
    지문, 질문, 선택지 등을 조합하여 모델에 입력할 프롬프트 템플릿을 생성하는 클래스.
    """

    SYSTEM_PROMPT_4_V1 = (
        "당신은 **지식 추론(Knowledge Inference) 전문가**입니다. "
        "이 유형은 정답이 지문에 그대로 쓰여 있지 않을 수 있으며, 지문은 '조건/단서'를 제공합니다. "
        "지문에서 주어진 조건을 정확히 반영하고, 그 조건과 모순되지 않는 범위에서 일반적으로 알려진 지식을 적용해 "
        "가장 타당한 선택지 하나를 고르십시오."
    )

    SYSTEM_PROMPT_5_V1 = (
        "당신은 논리적인 **텍스트 분석 및 독해 전문가**입니다. "
        "이 문제는 오직 **제공된 지문 내의 정보**만으로 풀어야 합니다. "
        "당신의 외부 배경지식을 배제하고, 철저하게 지문에 명시된 내용에 근거하여 판단하십시오.\n\n"
    )
    
    USER_PROMPT_PLUS_4_V1 = """### 지문
{paragraph}

### 질문
{question}

### 보기
{question_plus}

### 선택지
{choices}

### 문제 해결 가이드라인
1. 지문이 주는 조건/단서를 먼저 정리하세요. (무엇을 가정/설명하고 있는지)
2. 필요하면 일반적으로 알려진 지식(개념/원리/사례)을 적용하되, 지문 조건과 모순되면 안 됩니다.
3. 선택지 중 조건을 가장 잘 만족하는 것 하나만 고르세요.

정답은 1~4 중 하나의 정수로만 출력하세요. 다른 글자는 출력하지 마세요.
정답:"""


    # 4지선다 + <보기> 없음
    USER_PROMPT_NO_PLUS_4_V1 = """### 지문
{paragraph}

### 질문
{question}

### 선택지
{choices}

### 문제 해결 가이드라인
1. 지문이 주는 조건/단서를 먼저 정리하세요. (무엇을 가정/설명하고 있는지)
2. 필요하면 일반적으로 알려진 지식(개념/원리/사례)을 적용하되, 지문 조건과 모순되면 안 됩니다.
3. 선택지 중 조건을 가장 잘 만족하는 것 하나만 고르세요.

정답은 1~4 중 하나의 정수로만 출력하세요. 다른 글자는 출력하지 마세요.
정답:"""


    # 5지선다 + <보기> 있음
    USER_PROMPT_PLUS_5_V1 = """### 지문
{paragraph}

### 질문
{question}

### 보기
{question_plus}

### 선택지
{choices}

### 문제 해결 가이드라인
1. 지문을 끝까지 읽고 핵심 정보를 정리하세요.
2. 질문이 요구하는 정보(수치/인물/원인/결과/요지 등)가 무엇인지 정확히 확인하세요.
3. 각 선택지가 지문의 어느 부분과 일치하는지 1:1로 대조하세요.
4. 지문과 모순되거나 지문에 근거가 없는 선택지는 제외하세요.
5. 가장 확실한 근거를 가진 선택지 번호 하나만 선택하세요.

정답은 1~5 중 하나의 정수로만 출력하세요. 다른 글자는 출력하지 마세요.
정답:"""


    # 5지선다 + <보기> 없음
    USER_PROMPT_NO_PLUS_5_V1 = """### 지문
{paragraph}

### 질문
{question}

### 선택지
{choices}

### 문제 해결 가이드라인
1. 지문을 끝까지 읽고 핵심 정보를 정리하세요.
2. 질문이 요구하는 정보(수치/인물/원인/결과/요지 등)가 무엇인지 정확히 확인하세요.
3. 각 선택지가 지문의 어느 부분과 일치하는지 1:1로 대조하세요.
4. 지문과 모순되거나 지문에 근거가 없는 선택지는 제외하세요.
5. 가장 확실한 근거를 가진 선택지 번호 하나만 선택하세요.

정답은 1~5 중 하나의 정수로만 출력하세요. 다른 글자는 출력하지 마세요.
정답:"""

    @staticmethod
    def format_choices(choices: List[str]) -> str:
        """
        리스트 형태의 선택지를 번호와 함께 문자열로 포맷팅합니다.
        
        Args:
            choices: 선택지 텍스트들이 담긴 리스트
            
        Returns:
            "1 - 선택지" 형태로 줄바꿈된 전체 선택지 문자열
        """
        return "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)])
    
    def create_system_prompt(self, data: Dict) -> str:

        SYSTEM_PROMPTS = {
            "knowledge": {
                "v1": self.SYSTEM_PROMPT_4_V1,
            },
            "inferential": {
                "v1": self.SYSTEM_PROMPT_5_V1,
            }
        }

        SYSTEM_PROMPT_POLICY = {
            "knowledge": "v1",
            "inferential": "v1",
        }
        return SYSTEM_PROMPTS[data['question_type']][SYSTEM_PROMPT_POLICY[data['question_type']]]

    def create_user_prompt(self, data: Dict) -> str:
        """
        입력 데이터의 구성 요소에 따라 적절한 템플릿을 선택하여 최종 프롬프트를 생성합니다.
        
        Args:
            data: paragraph, question, choices, (선택적) question_plus를 포함하는 딕셔너리
            
        Returns:
            모델에 입력 가능한 완성된 프롬프트 문자열
        """
        choices_string = self.format_choices(data['choices'])
        USER_PROMPTS = {
        "knowledge": {
            "v1": {
                "plus": self.USER_PROMPT_PLUS_4_V1,
                "no_plus": self.USER_PROMPT_NO_PLUS_4_V1,
            },
            # "v2": {...}
        },
        "inferential": {
            "v1": {
                "plus": self.USER_PROMPT_PLUS_5_V1,
                "no_plus": self.USER_PROMPT_NO_PLUS_5_V1,
            },
            # "v2": {...}
            }
        }

        USER_PROMPT_POLICY = {
            "knowledge": "v1",
            "inferential": "v1",
        }

        template_set = USER_PROMPTS[data['question_type']][USER_PROMPT_POLICY[data['question_type']]]

        if data.get('question_plus'):
            return template_set['plus'].format(
                paragraph=data['paragraph'],
                question=data['question'],
                question_plus=data['question_plus'],
                choices=choices_string,
            )
        else:
            return template_set['no_plus'].format(
                paragraph=data['paragraph'],
                question=data['question'],
                choices=choices_string,
            )