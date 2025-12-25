import torch
from .inferential_model import InferentialModel
from .knowledge_model import KnowledgeModel

class DualModel:
    """
    문제 유형(선택지 개수)에 따라 추론형(5지) 또는 지식형(4지) 모델을 선택하여 예측을 수행하는 클래스.
    """
    
    def __init__(self, inferential_ckpt: str, knowledge_ckpt: str, device_map="auto"):
        """
        추론형 모델과 지식형 모델을 각각 로드하여 초기화합니다.
        
        Args:
            inferential_ckpt: 추론형 모델의 체크포인트 경로
            knowledge_ckpt: 지식형 모델의 체크포인트 경로
            device_map: 모델이 로드될 장치 설정 (기본값: "auto")
        """
        # 두 모델을 메모리에 로드 (VRAM 용량에 주의가 필요합니다)
        self.inferential = InferentialModel(inferential_ckpt, device_map=device_map)
        self.knowledge = KnowledgeModel(knowledge_ckpt, device_map=device_map)

    def predict(self, messages, len_choices, device="cuda"):
        """
        선택지 개수에 따라 적절한 모델을 라우팅하고, 마지막 토큰의 로짓을 기반으로 정답 확률을 계산합니다.
        
        Args:
            messages: 모델에 입력할 대화 형식의 메시지 리스트
            len_choices: 문제의 선택지 개수 (4개 또는 5개)
            device: 추론을 수행할 장치 (기본값: "cuda")
            
        Returns:
            각 선택지 번호(1~N)에 매핑된 확률값을 담은 numpy 배열
        """
        # 라우팅 로직 (선택지 개수에 따라 모델 및 토크나이저 선택)
        if len_choices == 5:
            target_wrapper = self.inferential
        else:
            target_wrapper = self.knowledge

        model = target_wrapper.get_model()
        tokenizer = target_wrapper.get_tokenizer()

        # 토크나이징 및 추론
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(input_ids)
            
        # 마지막 토큰의 로짓 추출 및 확률 변환
        logits = outputs.logits[:, -1].flatten().cpu()
        
        # 각 선택지 번호에 해당하는 보카 인덱스의 로짓값만 수집
        target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]
        
        # Softmax를 통해 확률 분포로 변환
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(target_logit_list, dtype=torch.float32), 
                dim=0
            )
            .detach()
            .cpu()
            .numpy()
        )
        
        return probs