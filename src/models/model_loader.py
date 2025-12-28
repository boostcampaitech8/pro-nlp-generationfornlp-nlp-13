import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

class ModelLoader:
    """
    HuggingFace의 AutoModel 클래스를 사용하여 Causal Language Model을 로드하는 클래스.
    """
    
    @staticmethod
    def load_model(
        model_name: str,
        torch_dtype=torch.float16,
        device_map: str = "auto",
        is_quantization: bool = False  # 양자화 여부 추가
    ):
        """
        Args:
            model_name: 로드할 모델의 이름 또는 허브 경로
            torch_dtype: 모델의 가중치 데이터 타입
            device_map: 장치 할당 전략
            is_quantization: True일 경우 4-bit 양자화 적용
        """
        
        quantization_config = None
        
        if is_quantization:
            # 4-bit NF4 양자화 설정 (가장 효율적인 설정)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,  # V100이면 fp16 추천,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quantization_config, # 설정 적용
        )
        
        if is_quantization:
            model = prepare_model_for_kbit_training(model)
        return model