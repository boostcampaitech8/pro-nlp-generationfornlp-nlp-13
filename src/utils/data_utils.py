import pandas as pd
from ast import literal_eval
from datasets import Dataset


PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""


def load_and_preprocess_data(data_path, inferential=False):
    """
    Load and preprocess the training data.
    
    Args:
        data_path: Path to the CSV file containing training data
        inferential: If True, filter for inferential (5-choice) problems.
                    If False, filter for knowledge (4-choice) problems.
        
    Returns:
        Dataset: Processed HuggingFace Dataset
    """
    # Load the train dataset
    dataset = pd.read_csv(data_path)
    
    # Flatten the JSON dataset
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
        }
        # Include 'question_plus' if it exists
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        else:
            record['question_plus'] = None
        records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Filter by problem type based on number of choices
    if inferential:
        # 5지선다 (추론형/독해형)
        df = df[df['choices'].apply(lambda x: len(x) == 5)]
        print(f"Filtered for inferential problems (5-choice): {len(df)} samples")
    else:
        # 4지선다 (지식형)
        df = df[df['choices'].apply(lambda x: len(x) == 4)]
        print(f"Filtered for knowledge problems (4-choice): {len(df)} samples")
    
    dataset = Dataset.from_pandas(df)
    
    # Process dataset into chat format
    processed_dataset = []
    for i in range(len(dataset)):
        choices_string = "\n".join([
            f"{idx + 1} - {choice}" 
            for idx, choice in enumerate(dataset[i]["choices"])
        ])
        
        # <보기>가 있을 때
        if dataset[i]["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                question_plus=dataset[i]["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                choices=choices_string,
            )
        
        # chat message 형식으로 변환
        processed_dataset.append({
            "id": dataset[i]["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"{dataset[i]['answer']}"}
            ],
            "label": dataset[i]["answer"],
        })
    
    processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))
    return processed_dataset


def formatting_prompts_func(example, tokenizer):
    """
    Format examples using the tokenizer's chat template.
    
    Args:
        example: Batch of examples
        tokenizer: HuggingFace tokenizer
        
    Returns:
        List of formatted text strings
    """
    output_texts = []
    for i in range(len(example["messages"])):
        output_texts.append(
            tokenizer.apply_chat_template(
                example["messages"][i],
                tokenize=False,
            )
        )
    return output_texts


def tokenize(element, tokenizer):
    """
    Tokenize a batch of elements.
    
    Args:
        element: Batch of examples
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    outputs = tokenizer(
        formatting_prompts_func(element, tokenizer),
        truncation=False,
        padding=False,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize entire dataset.
    
    Args:
        dataset: HuggingFace Dataset
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Tokenized dataset
    """
    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc="Tokenizing",
    )
    return tokenized_dataset