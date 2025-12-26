from trl import DataCollatorForCompletionOnlyLM
from transformers import PreTrainedTokenizer

class CollatorFactory:
    """
    DataCollator 생성 팩토리 클래스.
    """
    
    @staticmethod
    def create_completion_only_collator(
        tokenizer: PreTrainedTokenizer,
        response_template = "<|im_start|>assistant\n"
    ):
        """
        Assistant 응답 부분만 학습하는 DataCollatorForCompletionOnlyLM을 생성합니다.
        
        Args:
            tokenizer: 사용 중인 PreTrainedTokenizer 인스턴스
            response_template: 응답 시작을 알리는 템플릿 문자열
            
        Returns:
            학습 시 손실 계산을 응답 부분으로 제한하는 DataCollator 인스턴스
        """
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )