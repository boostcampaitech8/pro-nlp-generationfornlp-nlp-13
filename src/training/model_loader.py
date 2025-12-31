# src/training/model_loader.py
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-8B"
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: torch.dtype = torch.float16
    device_map: str = "auto"
    use_gradient_checkpointing: bool = True
    trust_remote_code: bool = True


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Union[str, List[str]] = "all-linear"
    adapter_path: Optional[str] = None  # resume/infer 시 사용
    bias: str = "none"
    task_type: TaskType = TaskType.CAUSAL_LM
    

def _get_bnb_config(model_cfg: ModelConfig) -> Optional[BitsAndBytesConfig]:
    if not model_cfg.use_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=model_cfg.compute_dtype,
        bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant
    )


def load_model(model_cfg: ModelConfig, lora_cfg: LoRAConfig):
    bnb_config = _get_bnb_config(model_cfg)

    print(f"Loading Base Model: {model_cfg.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
        use_cache=False if model_cfg.use_gradient_checkpointing else True,
    )

    if model_cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if model_cfg.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if lora_cfg is not None:
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
            task_type=lora_cfg.task_type,
        )

        model = get_peft_model(model, peft_config)
        print("Trainable Parameters:")
        model.print_trainable_parameters()
    
    return model


def load_model_inference(
        model_cfg: ModelConfig,
        adapter_path: str,
):
    bnb_config = _get_bnb_config(model_cfg)

    print(f"Loading Base Model for Inference: {model_cfg.model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,        
        use_cache=True, 
    )

    print(f"Loading LoRA Adapter from: {adapter_path}")
    
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False
    )

    model.eval()

    return model
