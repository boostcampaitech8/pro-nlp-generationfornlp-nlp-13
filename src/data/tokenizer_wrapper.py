from transformers import AutoTokenizer
from datasets import Dataset

class TokenizerWrapper:
    """Tokenizer 래퍼"""
    
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Tokenizer 초기 설정"""
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'
    
    def formatting_prompts_func(self, examples):
        """Chat 템플릿 적용"""
        output_texts = []
        for i in range(len(examples["messages"])):
            output_texts.append(
                self.tokenizer.apply_chat_template(
                    examples["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts
    
    def tokenize(self, element):
        """토큰화"""
        outputs = self.tokenizer(
            self.formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }
    
    def tokenize_dataset(self, dataset: Dataset, num_proc: int = 4) -> Dataset:
        """전체 Dataset 토큰화"""
        return dataset.map(
            self.tokenize,
            remove_columns=list(dataset.features),
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Tokenizing",
        )

