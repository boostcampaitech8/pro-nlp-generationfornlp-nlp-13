from typing import Dict, List

class PromptFormatter:
    """
    지문, 질문, 선택지 등을 조합하여 모델에 입력할 프롬프트 템플릿을 생성하는 클래스.
    """
    
    PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

    PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
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
    
    def create_prompt(self, data: Dict) -> str:
        """
        입력 데이터의 구성 요소에 따라 적절한 템플릿을 선택하여 최종 프롬프트를 생성합니다.
        
        Args:
            data: paragraph, question, choices, (선택적) question_plus를 포함하는 딕셔너리
            
        Returns:
            모델에 입력 가능한 완성된 프롬프트 문자열
        """
        choices_string = self.format_choices(data['choices'])
        
        if data.get('question_plus'):
            return self.PROMPT_QUESTION_PLUS.format(
                paragraph=data['paragraph'],
                question=data['question'],
                question_plus=data['question_plus'],
                choices=choices_string,
            )
        else:
            return self.PROMPT_NO_QUESTION_PLUS.format(
                paragraph=data['paragraph'],
                question=data['question'],
                choices=choices_string,
            )