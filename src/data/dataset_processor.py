from datasets import Dataset
import pandas as pd
from typing import List, Dict

class DatasetProcessor:
    """
    DataFrame의 데이터를 기반으로 HuggingFace Dataset을 생성하고 Chat 형식으로 변환하는 클래스.
    """
    
    def __init__(self, prompt_formatter):
        """
        DatasetProcessor 클래스를 초기화합니다.
        
        Args:
            prompt_formatter: 행 데이터를 텍스트 프롬프트로 변환하는 포매터 객체
        """
        self.prompt_formatter = prompt_formatter
    
    def to_chat_format(self, df: pd.DataFrame) -> List[Dict]:
        """
        DataFrame을 모델 학습을 위한 시스템/사용자/어시스턴트 메시지 형식으로 변환합니다.
        
        Args:
            df: 변환할 원본 데이터가 담긴 DataFrame
            
        Returns:
            학습용 Chat 메시지 구조를 가진 딕셔너리들의 리스트
        """
        processed_data = []
        
        for _, row in df.iterrows():
            user_message = self.prompt_formatter.create_prompt(row.to_dict())
            
            processed_data.append({
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"{row['answer']}"}
                ],
                "label": row["answer"],
            })
        
        return processed_data
    
    def to_test_format(self, df: pd.DataFrame) -> List[Dict]:
        """
        추론 및 평가를 위해 정답(assistant) 메시지를 제외하고 평가용 메타데이터를 추가합니다.
        
        Args:
            df: 변환할 테스트용 데이터가 담긴 DataFrame
            
        Returns:
            추론용 메시지 구조와 선택지 개수 정보(len_choices)가 포함된 리스트
        """
        processed_data = []
        for _, row in df.iterrows():
            user_message = self.prompt_formatter.create_prompt(row.to_dict())
            # choices 컬럼이 있다면 길이를 계산 (없으면 기본값 설정)
            len_choices = len(row["choices"]) if "choices" in row else 0
            
            processed_data.append({
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            })
        return processed_data
    
    def process(self, df: pd.DataFrame, is_test: bool = False) -> Dataset:
        """
        테스트 여부에 따라 적절한 포맷으로 변환 후 HuggingFace Dataset 객체를 반환합니다.
        
        Args:
            df: 처리할 원본 DataFrame
            is_test: True일 경우 테스트용 포맷으로, False일 경우 학습용 포맷으로 처리
            
        Returns:
            변환된 데이터를 포함하는 HuggingFace Dataset 객체
        """
        if is_test:
            chat_data = self.to_test_format(df)
        else:
            chat_data = self.to_chat_format(df)
        return Dataset.from_pandas(pd.DataFrame(chat_data))