import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig


def setup_model_and_tokenizer(
    model_name="Qwen/Qwen3-8B",
    lora_r=6,
    lora_alpha=8,
    lora_dropout=0.05,
):
    """
    Setup model, tokenizer, and PEFT config.
    
    Args:
        model_name: Name or path of the pretrained model
        lora_r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        
    Returns:
        Tuple of (model, tokenizer, peft_config)
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Setup pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    
    # Setup PEFT config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['q_proj', 'k_proj'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    return model, tokenizer, peft_config