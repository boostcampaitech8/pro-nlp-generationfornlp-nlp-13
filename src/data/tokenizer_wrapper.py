from dataclasses import dataclass
from typing import Any, Dict, Union


@dataclass(frozen=True)
class TokenizerConfig:
    max_length: int = 2048
    padding: Union[bool, str] = False
    truncation: bool = True
    add_generation_prompt: bool = False

class TokenizerWrapper:
    def __init__(self, tokenizer: Any, config: TokenizerConfig):
        self.tokenizer = tokenizer
        self.cfg = config

    def to_text(self, example: Dict[str, Any]) -> Dict[str, str]:
        text = self.tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=self.cfg.add_generation_prompt,
            enable_thinking=False,
        )

        return {"text": text}

    def tokenize_fn(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # batched=False -> str / batched=True -> List[str]
        texts = example["text"]
        if isinstance(texts, str):
            texts = [texts]

        tok_kwargs: Dict[str, Any] = {
            "truncation": self.cfg.truncation,
            "padding": self.cfg.padding,
        }
        if self.cfg.truncation:
            tok_kwargs["max_length"] = self.cfg.max_length
        out = self.tokenizer(texts, **tok_kwargs)

        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
        }