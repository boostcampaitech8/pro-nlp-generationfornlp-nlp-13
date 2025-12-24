import torch
from transformers import AutoModelForCausalLM

class ModelLoader:
    """모델 로딩"""
    
    @staticmethod
    def load_model(
        model_name: str,
        torch_dtype=torch.float16,
        device_map: str = "auto"
    ):
        """기본 모델 로딩"""
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )