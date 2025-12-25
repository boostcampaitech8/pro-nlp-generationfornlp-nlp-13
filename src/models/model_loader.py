import torch
from transformers import AutoModelForCausalLM

class ModelLoader:
    """
    HuggingFace의 AutoModel 클래스를 사용하여 Causal Language Model을 로드하는 클래스.
    """
    
    @staticmethod
    def load_model(
        model_name: str,
        torch_dtype=torch.float16,
        device_map: str = "auto"
    ):
        """
        사전 학습된 가중치를 로드하고 장치(GPU/CPU)에 할당합니다.
        
        Args:
            model_name: 로드할 모델의 이름 또는 허브 경로
            torch_dtype: 모델의 가중치 데이터 타입 (기본값: torch.float16)
            device_map: 모델 파라미터를 장치에 할당하는 전략 (기본값: "auto")
            
        Returns:
            로드된 AutoModelForCausalLM 인스턴스
        """
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )