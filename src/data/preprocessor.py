import pandas as pd
import ast


def parse_problems_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'problems' not in df.columns:
        return df

    df['problems'] = df['problems'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    problems_df = df['problems'].apply(pd.Series)
    df = df.drop(columns=['problems'])
    overlapping_cols = set(df.columns) & set(problems_df.columns)
    if overlapping_cols:
        df = df.drop(columns=list(overlapping_cols))

    df = pd.concat([df, problems_df], axis=1)
    
    return df

def add_choices_len(df: pd.DataFrame) -> pd.DataFrame:
    if 'choices' in df.columns:
        df['choices_len'] = df['choices'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    return df