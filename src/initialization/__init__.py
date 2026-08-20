from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from lightning import LightningModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.algorithm import create_algorithm
from src.algorithm.core.amber import Amber
from src.algorithm.dataloader import get_dataloader, get_datasets
from src.initialization.init_config import initialize_config
from src.initialization.init_seed import initialize_seed
from src.logger import CustomWandBLogger, get_wandb_logger


@dataclass
class InitializationReturn:
    """
    Data structure to store the results of the initialization process.

    Attributes:
        dataloaders (Dict[str, DataLoader]): A dictionary mapping dataset modes (e.g., "train", "val")
            to their corresponding PyTorch DataLoader instances.
        algorithm (Amber): The instantiated algorithm object based on the provided configuration.
        wandb_logger (Union[bool, CustomWandBLogger]): If Weights & Biases (WandB) logging is enabled,
            this will be an instance of CustomWandBLogger; otherwise, it will be False.
    """

    dataloaders: Dict[str, DataLoader]
    datasets: Dict[str, object]
    algorithm: LightningModule
    wandb_logger: bool | CustomWandBLogger


def initialize(*, config: DictConfig) -> InitializationReturn:
    """
    Initializes the training setup by setting up configurations, loading datasets,
    creating dataloaders, instantiating the algorithm, and configuring logging.

    Args:
        config (DictConfig): Configuration object containing experiment settings.

    Returns:
        InitializationReturn: A data structure containing the initialized dataloaders, algorithm,
        and WandB logger (if enabled).
    """
    initialize_config(config)  # Apply SLURM-related settings to the configuration
    initialize_seed(config.seed)  # Set random seed for reproducibility

    datasets = get_datasets(algorithm_config=config.algorithm, task_config=config.task)

    # Create dataloaders for each dataset mode (e.g., "train", "val", "test)
    dataloaders = {
        dataset_mode: get_dataloader(algorithm_config=config.algorithm, dataset=dataset, is_train=dataset_mode == "train")
        for dataset_mode, dataset in datasets.items()
    }

    # Instantiate the algorithm using the provided configuration
    algorithm = create_algorithm(algorithm_config=config.algorithm, train_dataset=datasets.get("train"))
    if hasattr(algorithm, "initialize_from_weighted_baseline_checkpoint_codex"):
        # [CodeX] 在 Trainer 接管前显式做一次 baseline checkpoint 迁移初始化，避免新旧结构不一致时 strict 加载失败。
        algorithm.initialize_from_weighted_baseline_checkpoint_codex()

    # Initialize WandB logger if enabled in the configuration
    if config.logger.wandb.enabled:
        wandb_logger = get_wandb_logger(config=config)
    else:
        wandb_logger = False
    return InitializationReturn(dataloaders, datasets, algorithm, wandb_logger)
