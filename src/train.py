import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import pandas as pd
import numpy as np
import random
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

from utils.arguments import parse_args
from utils.data_utils import load_and_preprocess_data, tokenize_dataset
from utils.model_utils import setup_model_and_tokenizer
from metrics import preprocess_logits_for_metrics, compute_metrics

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def main():
    # Parse arguments
    args = parse_args()
    
    # Determine problem type and output directory
    problem_type = "inferential" if args.inferential else "knowledge"
    choice_count = "5-choice" if args.inferential else "4-choice"
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"result/{problem_type}_model"
    
    # Create result directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training {problem_type.upper()} model ({choice_count})")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    processed_dataset = load_and_preprocess_data(
        args.data_path, 
        inferential=args.inferential
    )
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer, peft_config = setup_model_and_tokenizer(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(processed_dataset, tokenizer)
    
    # Filter by max length
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) <= args.max_seq_length
    )
    
    # Split dataset
    tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=args.test_size, 
        seed=args.seed
    )
    
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['test']
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Setup data collator
    # response_template = "<start_of_turn>model"
    response_template = "<|im_start|>assistant"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    
    # Setup training config
    sft_config = SFTConfig(
        do_train=True,
        do_eval=True,
        lr_scheduler_type=args.lr_scheduler_type,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.evaluation_strategy,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        report_to=args.report_to,
    )
    
    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        peft_config=peft_config,
        args=sft_config,
    )
    
    # Train
    print(f"\nStarting training for {problem_type} model...")
    trainer.train()
    
    print(f"\n{'='*60}")
    print(f"Training completed for {problem_type.upper()} model!")
    print(f"Model saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()