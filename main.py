# 标准库导入
# logging：日志记录
# os：操作系统交互
# sys：系统相关参数和函数
# traceback：常堆栈跟踪
# warnings：警告控制
# List：类型注解
import logging
import os
import sys
import traceback
import warnings
from typing import List

# 第三方库导入
# hydra：配置管理系统
# torch：PyTorch 深度学习框架
# LightningModule：Lightning 模型基类
# Trainer：Lightning 训练器
# LearningRateMonitor：学习率监控回调
# ModelCheckpoint：模型检查点回调
# DictConfig, OmegaConf：配置管理工具
import hydra
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

# 项目内部模块导入：
# initialize：初始化函数（核心初始化逻辑）
# load_omega_conf_resolvers：加载 OmegaConf 解析器
# CustomProgressBar：自定义进度条
from src.initialization import initialize
from src.initialization.init_config import load_omega_conf_resolvers
from src.logger.progress_bar import CustomProgressBar
from src.experiment_artifacts import (
    FormalLastCheckpointCallback,
    LocalMetricsCallback,
    initialize_run_artifacts,
    preflight_experiment_protocol,
)

# full stack trace,设置 Hydra 显示完整堆栈跟踪信息，便于调试。
os.environ["HYDRA_FULL_ERROR"] = "1"

# register OmegaConf resolver for hydra
# 加载 OmegaConf 解析器，支持在配置文件中使用自定义解析函数。
load_omega_conf_resolvers()

# 忽略 UserWarning 类别的警告，减少控制台输出干扰。
warnings.filterwarnings("ignore", category=UserWarning)

# 设置 skfem 库的日志级别为 WARNING，只显示警告和错误信息。
logging.getLogger("skfem").setLevel(logging.WARNING)  # Only show warnings and errors

# 使用 PyTorch Lightning 和 Hydra 的训练主程序。
# 装饰器：@hydra.main 指定 Hydra 配置
# version_base=None：不使用特定版本的 Hydra
# config_path="config"：配置文件所在目录
# config_name="training_config"：配置文件名（不包含扩展名）
@hydra.main(version_base=None, config_path="config", config_name="training_config")
def train(config: DictConfig) -> None:
    # 函数签名：接收一个 DictConfig 类型的配置对象
    try:
        logging.getLogger("skfem").setLevel(logging.WARNING)  # Only show warnings and errors
        print(OmegaConf.to_yaml(config, resolve=True))  # 打印解析后的完整配置（YAML格式），resolve=True 会解析配置中的所有变量

        # 获取 Hydra 运行时的输出目录，这是实验日志和检查点的保存位置
        exp_root = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        os.makedirs(exp_root, exist_ok=True)
        OmegaConf.save(config=config, f=os.path.join(exp_root, "resolved_config.yaml"), resolve=True)
        protocol_preflight = (
            preflight_experiment_protocol(config)
            if config.get("experiment_protocol")
            else None
        )

        # 如果配置中指定了矩阵乘法精度，设置 PyTorch 的浮点矩阵乘法精度；可能的值："high", "medium", "highest"
        if config.trainer.get("matmul_precision", None) is not None:
            torch.set_float32_matmul_precision(config.trainer.matmul_precision)

        # 核心初始化调用：调用 initialize() 函数，返回一个包含多个组件的对象
        initialization_return = initialize(config=config)
        if protocol_preflight is not None:
            initialize_run_artifacts(
                config=config,
                run_root=exp_root,
                initialization_return=initialization_return,
                preflight=protocol_preflight,
            )

        # 构建训练组件
        logger = initialization_return.wandb_logger # 从初始化返回中获取 WandB 日志记录器
        callbacks = get_callbacks(config, exp_root, logger, checkpoint_frequency=config.trainer.checkpoint_frequency) # 调用 get_callbacks() 函数获取回调列表
        trainer_config = config.trainer
        trainer_max_epochs = _resolve_max_epochs_for_stage2(config)
        trainer = Trainer(
            logger=logger,  # Use the wandb logger
            callbacks=callbacks,  # Checkpointing callback
            default_root_dir=exp_root,  # Where to save logs and checkpoints
            max_epochs=trainer_max_epochs,
            accelerator=trainer_config.accelerator,
            devices=trainer_config.devices,
            precision=trainer_config.precision,
            accumulate_grad_batches=trainer_config.accumulate_grad_batches,
            check_val_every_n_epoch=trainer_config.check_val_every_n_epoch,
            enable_checkpointing=trainer_config.enable_checkpointing,
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        # 开始训练
        dataloaders = initialization_return.dataloaders # 从初始化返回中获取数据加载器和算法模型
        algorithm: LightningModule = initialization_return.algorithm

        # 可选编译：如果配置启用了 torch_compile，使用 PyTorch 2.0+ 的编译功能优化性能
        if config.trainer.get("torch_compile", False):
            # Compile the algorithm for performance optimization
            # Note: This is a placeholder. The actual compilation method may vary.
            algorithm = torch.compile(algorithm)  # Todo: Test! 注释表明这是待测试的功能

        # 训练, 加载训练数据集和验证数据集
        trainer.fit(
            algorithm,
            train_dataloaders=dataloaders.get("train"),
            val_dataloaders=dataloaders.get("val"),
            ckpt_path=trainer_config.get("ckpt_path"),
        )
        if protocol_preflight is not None:
            # 正式协议固定使用本次训练明确落盘的 last.ckpt，而不是内存中的最终状态。
            last_checkpoint = os.path.join(exp_root, "checkpoints", "last.ckpt")
            if not os.path.exists(last_checkpoint):
                raise FileNotFoundError(f"Required formal evaluation checkpoint not found: {last_checkpoint}")
            trainer.test(
                algorithm,
                dataloaders=dataloaders.get("test"),
                ckpt_path=last_checkpoint,
            )
        else:
            trainer.test(algorithm, dataloaders=dataloaders.get("test"))

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

#  回调函数定义
def get_callbacks(config: DictConfig, exp_root: str, wandb_logger, checkpoint_frequency: int) -> List["Callback"]:
    print(exp_root)

    # 模型检查点回调：
    # dirpath：检查点保存路径（exp_root/checkpoints/）
    # filename：文件名格式（包含 epoch 编号，两位填充）
    # every_n_epochs：每 N 个周期保存一次
    # save_top_k=-1：保存所有检查点
    # save_last=True：额外保存最后一个模型（last.ckpt，停止训练的轮次），用于恢复训练
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_root, "checkpoints"),  # Directory to save checkpoints
        filename="checkpoint-{epoch:02d}",  # Checkpoint filename format
        every_n_epochs=checkpoint_frequency,  # Save checkpoint every K epochs
        save_top_k=-1,  # Save all checkpoints
        save_last=True,  # Optionally save the most recent model
    )
    # 初始化回调列表，包含自定义进度条
    callbacks = [CustomProgressBar(), LocalMetricsCallback(exp_root)]
    if config.get("experiment_protocol"):
        callbacks.append(FormalLastCheckpointCallback(exp_root))

    # 如果启用了检查点保存，添加检查点回调
    if config.trainer.enable_checkpointing:
        callbacks.append(checkpoint_callback)

    # 如果使用了 WandB 日志记录器，添加学习率监控回调
    if wandb_logger:
        learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(learning_rate_monitor)

    # 返回完整的回调列表
    return callbacks


def _resolve_max_epochs_for_stage2(config: DictConfig) -> int:
    # [CodeX] 若阶段二是从 checkpoint 恢复继续训练，则自动把 max_epochs 扩成“checkpoint 已训 epoch + stage2_epochs”。
    trainer_config = config.trainer
    max_epochs = int(trainer_config.max_epochs)
    weighted_imitation_config = config.algorithm.get("weighted_imitation") or {}
    ckpt_path = trainer_config.get("ckpt_path")
    if not ckpt_path or not weighted_imitation_config.get("stage2_enable", False):
        return max_epochs

    stage2_epochs = int(weighted_imitation_config.get("stage2_epochs", 0))
    if stage2_epochs <= 0:
        return max_epochs

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    return max(max_epochs, checkpoint_epoch + 1 + stage2_epochs)


if __name__ == "__main__":
    train()

'''
代码流程总结
开始
  │
  ├─ 1. 导入模块和设置环境
  │
  ├─ 2. 加载 Hydra 配置 (training_config.yaml)
  │
  ├─ 3. 初始化：
  │     ├─ 打印配置
  │     ├─ 获取输出目录
  │     ├─ 设置矩阵乘法精度
  │     └─ 调用 initialize() 获取：
  │         ├─ wandb_logger
  │         ├─ dataloaders (train/val/test)
  │         └─ algorithm (模型)
  │
  ├─ 4. 构建训练器：
  │     ├─ 获取回调列表
  │     ├─ 配置 Trainer 参数
  │     └─ 可选：torch.compile 编译模型
  │
  ├─ 5. 执行训练：
  │     ├─ trainer.fit() 训练和验证
  │     └─ trainer.test() 测试
  │
  ├─ 6. 异常处理
  │
  └─ 结束
'''

'''
关键特性
1、配置驱动：使用 Hydra 管理所有配置
2、模块化初始化：initialize() 函数封装了复杂的初始化逻辑
3、完整的训练流程：训练 → 验证 → 测试
4、丰富的回调系统：进度条、检查点、学习率监控
5、实验管理：自动化的日志和检查点保存
6、错误处理：完整的异常捕获和报告

项目架构体现
这个 main.py 体现了项目的整体架构设计：
1、配置层：Hydra + OmegaConf
2、训练层：PyTorch Lightning Trainer
3、业务逻辑层：initialize() 函数封装的算法、数据、日志等
4、工具层：自定义进度条、回调管理等
'''
