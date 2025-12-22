from ast import literal_eval
from datasets import Dataset, DatasetDict
import pandas as pd
import re

dataset = pd.read_csv('../data/train.csv') 

def main():
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            "question_plus": problems.get('question_plus', None),
        }
        # Include 'question_plus' if it exists
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        records.append(record)
    
    # 섞기
    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

    df_with_4choices = df[df["choices"].apply(len) == 4].copy()
    df_with_5choices = df[df["choices"].apply(len) == 5].copy()

    # eval/train split
    train4, val4, test4 = split_dataset(df_with_4choices)
    train5, val5, test5 = split_dataset(df_with_5choices)
    
    ds4 = DatasetDict({
    "train": Dataset.from_pandas(train4, preserve_index=False),
    "validation": Dataset.from_pandas(val4, preserve_index=False),
    "test": Dataset.from_pandas(test4, preserve_index=False),
    })
    
    ds5 = DatasetDict({
    "train": Dataset.from_pandas(train5, preserve_index=False),
    "validation": Dataset.from_pandas(val5, preserve_index=False),
    "test": Dataset.from_pandas(test5, preserve_index=False),
    })
    
    # push to hub
    ds4.push_to_hub(
        "yhkimmy/4_choices",
        private=True,
        token="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")
    ds5.push_to_hub(
        "yhkimmy/5_choices",
        private=True,
        token="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")

    #  불러올때
    # ds4 = load_dataset("yhkimmy/4_choices")
    # ds5 = load_dataset("yhkimmy/5_choices")

    # train4 = ds4["train"]
    # train5 = ds5["train"]

def split_dataset(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1):
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    main()