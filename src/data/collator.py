from trl import DataCollatorForCompletionOnlyLM
from transformers import PreTrainedTokenizer

class CollatorFactory:
    """DataCollator 생성 팩토리"""
    
    @staticmethod
    def create_completion_only_collator(
        tokenizer: PreTrainedTokenizer,
        response_template: str = "<start_of_turn>model"
    ):
        """Assistant 응답 부분만 학습하는 collator 생성"""
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )