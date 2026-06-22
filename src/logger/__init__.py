import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.logger.custom_wandb_logger import CustomWandBLogger, reset_wandb_env


def get_wandb_logger(config: DictConfig) -> CustomWandBLogger:
    """
    Create a wandb logger with the given config and algorithm.
    """
    # [CodeX] 仅在真正启用 WandB 记录器时才导入 wandb，避免关闭日志时仍被可选依赖阻塞训练/冒烟入口。
    import wandb

    reset_wandb_env()

    logger_config = config.logger
    wandb_config = logger_config.wandb

    project_name = wandb_config.get("project_name")
    environment_name = wandb_config.task_name

    if wandb_config.get("task_name") is not None:
        project_name = project_name + "_" + wandb_config.get("task_name")
    elif environment_name is not None:
        project_name = project_name + "_" + environment_name

    groupname = wandb_config.get("group_name")[-127:]
    job_type = wandb_config.get("job_type")[-63:]
    runname = job_type[-63:] + "_" + wandb_config.get("run_name")[-63:]

    tags = wandb_config.get("tags", [])
    if tags is None:
        tags = []

    if "idx" in config:
        runname = (runname + "_i" + str(config.idx))[-127:]
        job_type = (job_type + "_i" + str(config.idx))[-63:]
        tags.append("i" + str(config.idx))

    if "_version" in config:
        runname = (runname + "_v" + str(config._version))[-127:]
        job_type = (job_type + "_v" + str(config._version))[-63:]
        tags.append("v" + str(config._version))

    entity = wandb_config.get("entity")
    start_method = wandb_config.get("start_method")
    settings = wandb.Settings(start_method=start_method) if start_method is not None else None

    wandb_logger = CustomWandBLogger(
        config=OmegaConf.to_container(config, resolve=True),
        project=project_name,
        name=runname,
        group=groupname,
        tags=tags,
        entity=entity,
        settings=settings,
        job_type=job_type,
        log_model=False,
        save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )

    wandb_logger.experiment.log_code(
        root=".",
        include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml") or path.endswith(".yml"),
        exclude_fn=lambda path: "wandb" in path or "__pycache__" in path or ".git" in path,
    )

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    src_dir = os.path.join(project_root, "src")
    wandb_logger.experiment.log_code(
        root=src_dir,
        name="src",
        include_fn=lambda path: any(path.endswith(ext) for ext in [".py"]),
        exclude_fn=lambda path: any(exclude in path for exclude in ["wandb", "__pycache__", ".git", ".ipynb_checkpoints", "outputs", "logs"]),
    )

    return wandb_logger
