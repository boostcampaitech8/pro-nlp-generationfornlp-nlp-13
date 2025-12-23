from ast import literal_eval
from datasets import Dataset, DatasetDict
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from utils.arguments import parse_args

def main():
    # parse arguments
    args = parse_args()
    dataset = pd.read_csv(args.data_path)
    
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
    
    df = pd.DataFrame(records)

    df_with_4choices = df[df["choices"].apply(len) == 4].copy()
    df_with_5choices = df[df["choices"].apply(len) == 5].copy()

    # eval/train split
    train4, val4 = train_test_split(
    df_with_4choices, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train5, val5 = train_test_split(
    df_with_5choices, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    ds4 = DatasetDict({
    "train": Dataset.from_pandas(train4, preserve_index=False),
    "validation": Dataset.from_pandas(val4, preserve_index=False),
    })
    
    ds5 = DatasetDict({
    "train": Dataset.from_pandas(train5, preserve_index=False),
    "validation": Dataset.from_pandas(val5, preserve_index=False),
    })
    
    # push to hub
    ds4.push_to_hub(
        args.hf_dataset_with_4choices,
        private=True,
        token=args.hf_token)
    ds5.push_to_hub(
        args.hf_dataset_with_5choices,
        private=True,
        token=args.hf_token)
    
    #  how to get dataset
    # ds4 = load_dataset("yhkimmy/4_choices")
    # ds5 = load_dataset("yhkimmy/5_choices")

if __name__ == "__main__":
    main()