from datasets import Dataset
import pandas as pd
from typing import List, Dict

class DatasetProcessor:
    """Dataset을 chat 형식으로 변환"""
    
    def __init__(self, prompt_formatter):
        self.prompt_formatter = prompt_formatter
    
    def to_chat_format(self, df: pd.DataFrame) -> List[Dict]:
        """DataFrame을 chat message 형식으로 변환"""
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
    
    def process(self, df: pd.DataFrame) -> Dataset:
        """전체 처리 파이프라인"""
        chat_data = self.to_chat_format(df)
        return Dataset.from_pandas(pd.DataFrame(chat_data))