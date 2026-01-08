import pandas as pd
from ast import literal_eval
from typing import Dict, List, Any
from torch.utils.data import Dataset
from prompt.prompt_formatter import create_system_prompt, create_user_prompt

class CustomDataSet(Dataset):
    def __init__(self, csv_path: str, tokenizer: Any, is_train: bool = True) -> None:
        df = pd.read_csv(csv_path)
        self.records = []

        for _, row in df.iterrows():
            problems = literal_eval(row['problems'])
            self.records.append({
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                'question_plus': problems.get('question_plus', None),
            })

        self.tokenizer = tokenizer
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        item = self.records[idx]
        
        messages = [
            {"role": "system", "content": create_system_prompt(self.to_dict())},
            {"role": "user", "content": create_user_prompt(self.to_dict())},
        ]

        if self.is_train:
            messages.append({"role": "assistant", "content": f"{item['answer']}"})
        
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            truncation=True,
            max_length=4096, 
            return_dict=True
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }