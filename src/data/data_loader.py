import pandas as pd
from ast import literal_eval
from typing import Dict, List

class DataLoader:
    """원본 CSV 데이터를 로드하고 파싱"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        
    def load(self) -> pd.DataFrame:
        """CSV 로드"""
        return pd.read_csv(self.csv_path)
    
    def flatten_problems(self, df: pd.DataFrame) -> pd.DataFrame:
        """problems JSON 필드를 flatten"""
        records = []
        for _, row in df.iterrows():
            problems = literal_eval(row['problems'])
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                'question_plus': problems.get('question_plus', None),
            }
            if 'question_plus' in problems:
                record['question_plus'] = problems['question_plus']
            records.append(record)
        return pd.DataFrame(records)
    
    def load_and_flatten(self) -> pd.DataFrame:
        """로드 + flatten 한번에"""
        df = self.load()
        return self.flatten_problems(df)
