import pandas as pd
from ast import literal_eval
from typing import Dict, List

class DataLoader:
    """
    원본 CSV 데이터를 로드하고 JSON 형식의 필드를 파싱하는 클래스.
    """
    
    def __init__(self, csv_path: str):
        """
        DataLoader 클래스를 초기화합니다.
        
        Args:
            csv_path: 로드할 CSV 파일의 경로
        """
        self.csv_path = csv_path
        
    def load(self) -> pd.DataFrame:
        """
        지정된 경로에서 CSV 파일을 로드합니다.
        
        Returns:
            로드된 원본 데이터를 담고 있는 pandas DataFrame
        """
        return pd.read_csv(self.csv_path)
    
    def flatten_problems(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        'problems' JSON 문자열 필드를 개별 컬럼으로 펼쳐서 새로운 DataFrame을 생성합니다.
        
        Args:
            df: 'problems' 컬럼을 포함하는 원본 DataFrame
            
        Returns:
            id, paragraph, question, choices 등이 개별 컬럼으로 분리된 DataFrame
        """
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
        """
        데이터 로드와 파싱(flatten) 과정을 한 번에 수행합니다.
        
        Returns:
            전처리가 완료된 학습용 DataFrame
        """
        df = self.load()
        return self.flatten_problems(df)