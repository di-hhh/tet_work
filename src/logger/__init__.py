import hydra # 导入 hydra 库，这是一个用于配置管理的框架
import wandb # 导入 Weights & Biases (wandb) 库，用于实验跟踪和可视化。
from omegaconf import DictConfig, OmegaConf # 用于处理结构化配置。
import os
from src.logger.custom_wandb_logger import CustomWandBLogger, reset_wandb_env

# 该函数创建一个基于给定配置的 wandb 日志记录器。
def get_wandb_logger(config: DictConfig) -> CustomWandBLogger:
    """
    Create a wandb logger with the given config and algorithm.
    Args:
        logger_config:

    Returns: A wandb logger to use. 返回 CustomWandBLogger 实例
    """
    reset_wandb_env()   # 重置 wandb 环境变量，以确保每次运行时都使用新的环境变量

    logger_config = config.logger # 获取配置文件中的 logger 部分，即logger/default_logger.yaml
    wandb_config = logger_config.wandb # 获取 wandb 相关的配置

    project_name = wandb_config.get("project_name") # 获取项目名称
    environment_name = wandb_config.task_name # 获取任务名称

    if wandb_config.get("task_name") is not None: # 如果任务名称不为空
        project_name = project_name + "_" + wandb_config.get("task_name") # 将任务名称添加到项目名称中
    elif environment_name is not None:
        project_name = project_name + "_" + environment_name
    else:
        # no further specification of the project, just use the initial project_name
        project_name = project_name

    groupname = wandb_config.get("group_name")[-127:] # 获取组名并限制长度为最多 127 个字符。
    job_type = wandb_config.get("job_type")[-63:]
    runname = job_type[-63:] + "_" + wandb_config.get("run_name")[-63:] # 获取运行名称

    tags = wandb_config.get("tags", [])
    if tags is None:
        tags = []

    if "idx" in config: # 如果配置中有索引(idx)，则更新运行名称、作业类型并在标签中添加idx标记（区分实验类型）。
        runname = (runname + "_i" + str(config.idx))[-127:]
        job_type = (job_type + "_i" + str(config.idx))[-63:]
        tags.append("i" + str(config.idx))

    if "_version" in config: # 如果配置中有版本号，则更新运行名称、作业类型并在标签中添加版本号标记。
        runname = (runname + "_v" + str(config._version))[-127:]
        job_type = (job_type + "_v" + str(config._version))[-63:]
        tags.append("v" + str(config._version))

    entity = wandb_config.get("entity")

    start_method = wandb_config.get("start_method")
    # 如果指定了启动方法，则创建相应的 wandb.Settings 对象。
    settings = wandb.Settings(start_method=start_method) if start_method is not None else None

    # 创建 wandb 日志记录器, 初始化自定义的 CustomWandBLogger 实例
    wandb_logger = CustomWandBLogger(
        config=OmegaConf.to_container(config, resolve=True), # 将 OmegaConf 配置转换为普通容器对象
        project=project_name,  # Name of your WandB project # WandB项目名称
        name=runname,  # Name of the current run            # 运行名称
        group=groupname,  # Group name for the run          # 运行组名称
        tags=tags,  # List of tags for your run             # 标签列表
        entity=entity,  # WandB username or team name
        settings=settings,  # Optional WandB settings
        job_type=job_type,  # Name of your experiment
        log_model=False,
        save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )

    # 记录整个源码文件夹
    wandb_logger.experiment.log_code(
        root=".",  # 当前目录
        include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml") or path.endswith(".yml"),
        exclude_fn=lambda path: "wandb" in path or "__pycache__" in path or ".git" in path
    )
    # 获取项目根目录（假设 src 在项目根目录下）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    src_dir = os.path.join(project_root, "src")

    # 记录整个 src 文件夹
    wandb_logger.experiment.log_code(
        root=src_dir,  # 指定 src 目录
        name="src",  # 为源码集合命名
        include_fn=lambda path: any(
            path.endswith(ext) for ext in [".py"]
        ),
        exclude_fn=lambda path: any(
            exclude in path for exclude in ["wandb", "__pycache__", ".git", ".ipynb_checkpoints", "outputs", "logs"]
        )
    )

    return wandb_logger