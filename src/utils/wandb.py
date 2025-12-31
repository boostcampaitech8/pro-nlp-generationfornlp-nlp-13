from typing import Any, Dict, Optional

import wandb

def wandb_init(
        config: Dict[str, Any],
        project_name: str="my_project",
        entity: str="pro-nlp-generationfornlp-nlp-13",
        run_name: Optional[str] = None,
    ):
    wandb.init(
        project=project_name,  # 프로젝트명 (원하는 대로 변경 가능)
        entity="pro-nlp-generationfornlp-nlp-13",  # 팀명
        name=run_name,
        config=config
    )

def wandb_finish() -> None:
    if wandb.run is not None:
        wandb.finish()