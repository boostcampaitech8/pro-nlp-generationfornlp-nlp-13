from peft import LoraConfig

class LoraConfigFactory:
    """LoRA 설정 생성"""
    
    @staticmethod
    def create_default_config(
        r: int = 6,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        target_modules: list = None
    ) -> LoraConfig:
        """기본 LoRA Config"""
        if target_modules is None:
            target_modules = ['q_proj', 'k_proj']
        
        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    @staticmethod
    def create_knowledge_config() -> LoraConfig:
        """4지선다(지식형)용 LoRA Config - 더 많은 파라미터"""
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
        """5지선다(독해형)용 LoRA Config"""
        return LoraConfig(
            r=6,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )