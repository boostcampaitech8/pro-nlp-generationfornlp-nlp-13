from typing import List, Tuple
import pandas as pd

class QuestionClassifier:
    """
    4지선다(지식형) vs 5지선다(추론형) 분류 클래스.
    """
    
    @staticmethod
    def classify(choices: List[str]) -> str:
        """
        선택지의 개수를 기준으로 문제 유형을 분류합니다.
        
        Args:
            choices: 문제의 선택지 리스트
            
        Returns:
            선택지 개수에 따른 분류 결과 ("knowledge" 또는 "inferential")
        """
        num_choices = len(choices)
        if num_choices == 4:
            return "knowledge"  # 4지선다 = 지식형
        elif num_choices == 5:
            return "inferential"  # 5지선다 = 추론형
        else:
            raise ValueError(f"Unexpected number of choices: {num_choices}")
    
    def classify_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 데이터셋에 대해 문제 유형 분류를 수행합니다.
        
        Args:
            df: 'choices' 컬럼을 포함하는 pandas DataFrame
            
        Returns:
            'question_type' 컬럼이 추가된 DataFrame
        """
        df['question_type'] = df['choices'].apply(self.classify)
        return df
    
    def split_by_type(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        데이터셋을 지식형(4지)과 추론형(5지)으로 분리합니다.
        
        Args:
            df: 분류할 원본 DataFrame
            
        Returns:
            (knowledge_df, inferential_df) 형태의 튜플
        """
        df = self.classify_dataset(df)
        knowledge_df = df[df['question_type'] == 'knowledge'].reset_index(drop=True)
        inferential_df = df[df['question_type'] == 'inferential'].reset_index(drop=True)
        return knowledge_df, inferential_df