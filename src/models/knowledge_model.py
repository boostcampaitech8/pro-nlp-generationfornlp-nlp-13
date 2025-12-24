import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

class KnowledgeModel:
    def __init__(self, checkpoint_path: str, device_map="auto"):
        print(f"[KnowledgeModel] Loading from {checkpoint_path}...")
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
        self.model.eval()

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer