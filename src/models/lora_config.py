from peft import LoraConfig

class LoraConfigFactory:
    """
    모델 타입(지식형/독해형)에 최적화된 LoRA 설정을 생성하는 팩토리 클래스.
    """

    @staticmethod
    def create_default_config(**kwargs) -> LoraConfig:
        """
        기본 설정을 바탕으로 하되, 입력받은 인자로 값을 덮어씌워 LoRA 설정을 생성합니다.
        
        Args:
            **kwargs: r, lora_alpha, target_modules 등 기본값을 변경할 파라미터들
            
        Returns:
            업데이트된 파라미터가 적용된 LoraConfig 인스턴스
        """
        # 기본값 설정
        config_params = {
            "r": 6,
            "lora_alpha": 8,
            "lora_dropout": 0.05,
            "target_modules": ['q_proj', 'k_proj'],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        
        # 입력받은 인자(kwargs)가 있다면 기본값을 덮어씌움
        config_params.update(kwargs)
        
        return LoraConfig(**config_params)
    
    @staticmethod
    def create_knowledge_config() -> LoraConfig:
        """
        4지선다(지식형) 학습에 최적화된 LoRA 설정을 생성합니다. (더 많은 모듈 타겟팅)
        
        Returns:
            지식 학습을 위해 확장된 target_modules를 포함하는 LoraConfig
        """
        return LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    @staticmethod
    def create_comprehension_config() -> LoraConfig:
        """
        5지선다(독해형) 학습에 최적화된 기본 LoRA 설정을 생성합니다.
        
        Returns:
            독해 문제에 적합한 표준 파라미터의 LoraConfig
        """
        return LoraConfig(
            r=6,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )