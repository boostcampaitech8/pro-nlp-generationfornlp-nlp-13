from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from src.data.preprocessor import parse_problems_column, add_choices_len
from src.prompt.prompt_builder import PromptBuilder, PromptConfig
from src.data.tokenizer_wrapper import TokenizerWrapper, TokenizerConfig

@dataclass(frozen=True)
class DataConfig:
    train_path: Optional[Union[str, Path]] = None
    test_path: Optional[Union[str, Path]] = None
    valid_ratio: float = 0.1
    seed: int = 42

    do_split: bool = True


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path)


def build_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_problems_column(df)
    df = add_choices_len(df)

    return df


def df_to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def make_train_valid_dataset(
    data_cfg: DataConfig,
    prompt_cfg: PromptConfig,
    tokenize_cfg_train: TokenizerConfig,
    tokenize_cfg_gen: TokenizerConfig,
    tokenizer,
) -> DatasetDict:
    
    df = load_csv(data_cfg.train_path)
    df = build_dataframe(df)

    if data_cfg.do_split:
        train_df, valid_df = train_test_split(
            df,
            test_size=data_cfg.valid_ratio,
            stratify=df["choices_len"],
            random_state=data_cfg.seed,
        )
        train_ds = Dataset.from_pandas(train_df, preserve_index=False)
        valid_ds = Dataset.from_pandas(valid_df, preserve_index=False)
    else:
        train_ds = Dataset.from_pandas(df, preserve_index=False)
        valid_ds = None
    
    prompt_cfg_train = PromptConfig(
        policy=prompt_cfg.policy,
        mode="train",
        templates_dir=prompt_cfg.templates_dir,
        verbose=prompt_cfg.verbose,
    )
    prompt_cfg_test = PromptConfig(
        policy=prompt_cfg.policy,
        mode="test",
        templates_dir=prompt_cfg.templates_dir,
        verbose=prompt_cfg.verbose,
    )

    builder_train = PromptBuilder(prompt_cfg_train)
    builder_test = PromptBuilder(prompt_cfg_test)

    tokenize_wrapper_train = TokenizerWrapper(tokenizer, tokenize_cfg_train)
    tokenize_wrapper_gen = TokenizerWrapper(tokenizer, tokenize_cfg_gen)

    train_msg = train_ds.map(
        builder_train.build_message,
        batched=False,
        remove_columns=train_ds.column_names,
        desc="Build train messages",
    )
    train_text = train_msg.map(
        tokenize_wrapper_train.to_text,
        batched=False,
        remove_columns=["messages"],
        desc="Serialize train to text",
    )
    
    if valid_ds is None:
        return DatasetDict({"train": train_text})

    valid_msg = valid_ds.map(
        builder_train.build_message,
        batched=False,
        remove_columns=valid_ds.column_names,
        desc="Build valid messages (teacher forcing)",
    )
    valid_text = valid_msg.map(
        tokenize_wrapper_train.to_text,
        batched=False,
        remove_columns=["messages"],
        desc="Serialize valid to text",
    )

    valid_gen_msg = valid_ds.map(
        builder_test.build_message,
        batched=False,
        desc="Build valid_gen messages (prompt only)",
    )

    # 이게 맞는건지 모르겠음. -> 일단 보류
    def _to_text_keep_meta(ex):
        out = tokenize_wrapper_gen.to_text(ex)  
        out["id"] = ex["id"]
        out["answer"] = ex["answer"]
        out["choices_len"] = ex["choices_len"]
        out["choices"] = ex["choices"]
        return out

    valid_gen_text = valid_gen_msg.map(
        _to_text_keep_meta,
        batched=False,
        # message 
        remove_columns=["messages"],
        desc="Serialize valid_gen to text (+meta)",
    )

    return DatasetDict(
        {
            "train": train_text,
            "validation": valid_text,
            "validation_gen": valid_gen_text,
        }
    )