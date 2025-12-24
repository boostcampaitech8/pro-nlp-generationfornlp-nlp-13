import torch
from .inferential_model import InferentialModel
from .knowledge_model import KnowledgeModel

class DualModel:
    def __init__(self, inferential_ckpt: str, knowledge_ckpt: str, device_map="auto"):
        # 두 모델을 메모리에 로드 (VRAM 주의)
        self.inferential = InferentialModel(inferential_ckpt, device_map=device_map)
        self.knowledge = KnowledgeModel(knowledge_ckpt, device_map=device_map)

    def predict(self, messages, len_choices, device="cuda"):
        """
        len_choices에 따라 적절한 모델과 토크나이저를 선택하여 로짓을 반환
        """
        # 1. 라우팅 로직 (Classifier 역할 통합)
        if len_choices == 5:
            target_wrapper = self.inferential
        else:
            target_wrapper = self.knowledge

        model = target_wrapper.get_model()
        tokenizer = target_wrapper.get_tokenizer()

        # 2. 토크나이징 및 추론
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(input_ids)
            
        # 3. 마지막 토큰의 로짓 반환 (logits, tokenizer 반환하여 후처리 위임)
        logits = outputs.logits[:, -1].flatten().cpu()
        return logits, tokenizer