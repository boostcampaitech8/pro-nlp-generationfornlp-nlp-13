from typing import List
import pandas as pd

class QuestionClassifier:
    """4지선다(지식형) vs 5지선다(독해형) 분류"""
    
    @staticmethod
    def classify(choices: List[str]) -> str:
        """선택지 개수로 분류"""
        num_choices = len(choices)
        if num_choices == 4:
            return "knowledge"  # 4지선다 = 지식형
        elif num_choices == 5:
            return "comprehension"  # 5지선다 = 독해형
        else:
            raise ValueError(f"Unexpected number of choices: {num_choices}")
    
    def classify_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 데이터셋 분류"""
        df['question_type'] = df['choices'].apply(self.classify)
        return df
    
    def split_by_type(self, df: pd.DataFrame) -> tuple:
        """4지/5지 선다로 분리"""
        df = self.classify_dataset(df)
        knowledge_df = df[df['question_type'] == 'knowledge'].reset_index(drop=True)
        comprehension_df = df[df['question_type'] == 'comprehension'].reset_index(drop=True)
        return knowledge_df, comprehension_df