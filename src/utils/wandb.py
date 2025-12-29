import wandb

def wandb_init(config, project_name="my_project", run_name=None):
    # WandB 초기화
    wandb.init(
        project=project_name,  # 프로젝트명 (원하는 대로 변경 가능)
        entity="pro-nlp-generationfornlp-nlp-13",  # 팀명
        name=run_name,
        config=config
    )

def wandb_finish():
    if wandb.run is not None:
        wandb.finish()