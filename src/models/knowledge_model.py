import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

class KnowledgeModel:
    """
    지식형(4지선다) 문제를 해결하기 위해 양자화된 LoRA 모델과 토크나이저를 관리하는 클래스.
    """
    
    def __init__(self, checkpoint_path: str, device_map="auto"):
        """
        지정된 체크포인트로부터 8비트 양자화된 지식형 모델과 토크나이저를 로드합니다.
        
        Args:
            checkpoint_path: PEFT 체크포인트가 저장된 경로
            device_map: 모델 파라미터를 장치에 할당하는 전략 (기본값: "auto")
        """
        print(f"[KnowledgeModel] Loading from {checkpoint_path}...")

        # 8-bit 양자화 설정 (추론형 모델과 동일한 효율적 메모리 관리 적용)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True # CPU 오프로드 시 안정성 확보
        )
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
        self.model.eval()

    def get_model(self):
        """
        현재 로드된 지식형 모델 인스턴스를 반환합니다.
        
        Returns:
            AutoPeftModelForCausalLM 모델 객체
        """
        return self.model

    def get_tokenizer(self):
        """
        현재 로드된 모델에 대응하는 토크나이저 인스턴스를 반환합니다.
        
        Returns:
            AutoTokenizer 객체
        """
        return self.tokenizer