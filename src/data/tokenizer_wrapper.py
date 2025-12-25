from transformers import AutoTokenizer
from datasets import Dataset

class TokenizerWrapper:
    """
    AutoTokenizer를 래핑하여 모델에 최적화된 설정 및 데이터셋 토크나이징을 수행하는 클래스.
    """
    
    def __init__(self, model_name: str):
        """
        TokenizerWrapper 클래스를 초기화하고 지정된 모델의 토크나이저를 로드합니다.
        
        Args:
            model_name: 로드할 모델의 이름 또는 경로
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """
        토크나이저의 PAD 토큰 설정 및 패딩 방향 등 초기 설정을 수행합니다.
        """
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'
    

    def formatting_prompts_func(self, example):
        """
        토크나이저의 chat template을 사용하여 예시 데이터들을 텍스트 형식으로 포맷팅합니다.
        
        Args:
            example: 'messages' 키를 포함하는 배치 데이터
            
        Returns:
            Chat 템플릿이 적용된 포맷팅된 텍스트 문자열 리스트
        """
        output_texts = []
        for i in range(len(example["messages"])):
            # Qwen3 모델의 경우 enable_thinking=False 설정하여 <think> 태그 제거
            output_texts.append(
                self.tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                    enable_thinking=False,  # 핵심: thinking 모드 비활성화
                )
            )
        return output_texts

    
    def tokenize(self, element):
        """
        배치 데이터에 대해 토크나이징을 수행합니다.
        
        Args:
            element: 포맷팅할 문장들이 포함된 배치 데이터
            
        Returns:
            input_ids와 attention_mask를 포함하는 딕셔너리
        """
        outputs = self.tokenizer(
            self.formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }
    
    def tokenize_dataset(self, dataset: Dataset, num_proc: int = 4) -> Dataset:
        """
        전체 데이터셋에 대해 병렬 처리를 사용하여 토크나이징을 적용합니다.
        
        Args:
            dataset: 토크나이징을 수행할 원본 HuggingFace Dataset
            num_proc: 병렬 처리에 사용할 프로세스 수
            
        Returns:
            토크나이징이 완료되고 기존 컬럼이 제거된 새로운 Dataset 객체
        """
        return dataset.map(
            self.tokenize,
            remove_columns=list(dataset.features),
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Tokenizing",
        )