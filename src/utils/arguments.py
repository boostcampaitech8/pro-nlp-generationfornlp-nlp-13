import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Korean language model for reading comprehension")
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/ephemeral/pro-nlp-generationfornlp-nlp-13/data/train.csv",
        help="Path to the training data CSV file"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proportion of data to use for validation"
    )
    parser.add_argument(
        "--inferential",
        action="store_true",
        help="If set, train on inferential (5-choice) problems. Otherwise, train on knowledge (4-choice) problems"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length for training"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=6,
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (default: auto-generated based on problem type)"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device for training"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size per device for evaluation"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X steps"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="Checkpoint save strategy"
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        help="Evaluation strategy"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Reporting destination (wandb, tensorboard, none)"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    return args