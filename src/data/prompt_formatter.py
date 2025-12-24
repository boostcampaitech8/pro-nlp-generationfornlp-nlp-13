from typing import Dict, List

class PromptFormatter:
    """프롬프트 템플릿 생성"""
    
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
        """선택지를 문자열로 포맷팅"""
        return "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)])
    
    def create_prompt(self, data: Dict) -> str:
        """데이터로부터 프롬프트 생성"""
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