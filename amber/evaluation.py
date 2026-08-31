import logging
import os
import sys
import traceback
import warnings
from typing import Dict, List

# deterministic cublas implementation (for reproducibility)
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import hydra
import numpy as np
import torch
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
# hydra 用于配置管理，torch 用于深度学习，lightning 用于训练，omegaconf 用于结构化配置，tqdm 用于进度条。

# 导入项目内部模块：
# create_algorithm：创建算法模型。
# get_dataloader 和 get_datasets：获取数据集和数据加载器。
# initialize_seed：设置随机种子。
# initialize_config：初始化配置。
# load_omega_conf_resolvers：注册 OmegaConf 解析器。
# EvaluationLogger：评估日志记录器。
from src.algorithm import create_algorithm
from src.algorithm.dataloader import get_dataloader, get_datasets
from src.initialization import initialize_seed
from src.initialization.init_config import initialize_config, load_omega_conf_resolvers
from src.logger.evaluation_logger import EvaluationLogger

# full stack trace, 设置 Hydra 显示完整堆栈信息。
os.environ["HYDRA_FULL_ERROR"] = "1"

# register OmegaConf resolver for hydra，注册 OmegaConf 解析器。
load_omega_conf_resolvers()

# 忽略 UserWarning 类警告。
warnings.filterwarnings("ignore", category=UserWarning)

# 设置 skfem 日志级别为 WARNING。
logging.getLogger("skfem").setLevel(logging.WARNING)  # Only show warnings and errors


# 使用 hydra.main 装饰器，指定配置文件路径为 config/test_config.yaml。
# 函数接收一个 DictConfig 类型的配置对象。
@hydra.main(version_base=None, config_path="config", config_name="test_config")
def evaluation(config: DictConfig) -> None:
    try:
        # 打印解析后的完整配置（YAML格式），resolve=True 会解析配置中的所有变量
        print(OmegaConf.to_yaml(config, resolve=True))

        # 获取 Hydra 运行时的输出目录路径。
        exp_root = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        initialize_config(config)  # Apply SLURM-related settings to the configuration，初始化配置（如SLURM集群相关设置）。
        initialize_seed(config.seed)  # Set random seed for reproducibility，设置随机种子以保证结果可重现。

        # 获取数据集（训练、验证、测试）。
        datasets = get_datasets(algorithm_config=config.algorithm, task_config=config.task) # 卡在这里

        # Create dataloaders for each dataset mode (e.g., "train", "val", "test)
        # 为每个数据集模式（如 train、val、test）创建数据加载器。
        dataloaders = {
            dataset_mode: get_dataloader(algorithm_config=config.algorithm, dataset=dataset, is_train=dataset_mode == "train")
            for dataset_mode, dataset in datasets.items()
        }

        # 如果配置中指定了矩阵乘法精度，则设置 PyTorch 的浮点矩阵乘法精度。
        if config.trainer.get("matmul_precision", None) is not None:
            torch.set_float32_matmul_precision(config.trainer.matmul_precision)

        loading_path = config.loading.root_path
        if loading_path.endswith("/"):
            loading_path = loading_path[:-1] # 去除加载路径末尾的斜杠。
        output_path = config.loading.output_path
        os.makedirs(output_path, exist_ok=True) # 创建输出目录。
        # last folder is exp_name
        exp_name = loading_path.split("/")[-1] # 从加载路径中提取实验名称。

        # 调用 get_ckpts 函数获取要加载的检查点队列。
        ckpt_queue = get_ckpts(loading_path, checkpoint_loading=config.loading.checkpoint)

        # 检查点循环评估
        # 遍历检查点队列，每个检查点包含：路径、任务类型、随机种子
        for ckpt_dict in tqdm(ckpt_queue, desc="Evaluating checkpoints", unit="checkpoint"):
            checkpoint_path = ckpt_dict["checkpoint_path"]
            job_type = ckpt_dict["job_type"]
            seed = ckpt_dict["seed"]

            # 创建算法模型，并加载指定的检查点。
            algorithm = create_algorithm(
                algorithm_config=config.algorithm,
                train_dataset=datasets.get("train"),
                loading=True,
                checkpoint_path=checkpoint_path,
            )

            # this is the correct parent directory to load the algorithm and the checkpoints
            # 创建评估日志记录器。
            test_logger = EvaluationLogger(
                output_path=config.loading.output_path, exp_name=exp_name, save_figures=config.loading.save_figures, job_type=job_type, seed=seed
            )

            # 创建了一个 PyTorch Lightning 的 Trainer 对象
            trainer = Trainer(
                logger=test_logger,  # results will be written to disk
                max_epochs=config.trainer.max_epochs,  # Max number of epochs for training
                accelerator=config.trainer.accelerator,  # what type of accelerator to use
                devices=config.trainer.devices,  # how many devices to use (if accelerator is not None)
                precision=config.trainer.precision,  # Precision setting (e.g., 16-bit)
                callbacks=None,
                default_root_dir=exp_root,  # dummy path, will be used multiple times presumably
                enable_checkpointing=False,  # no checkpointing
            )
            # 允许修改 sizing_field_damping 的属性
            OmegaConf.set_struct(algorithm.sizing_field_damping, False)

            # 获取完整的阻尼配置对象
            last_step_damping_config = config._evaluations.last_step_damping # 此处原来少了一个_evaluations
            if last_step_damping_config.get("do_last_step_damping"):  # 检查配置中是否启用了阻尼参数扫描
                # 保存原始最小尺寸场
                # 存原始值，以便后续基于此值进行缩放
                # gmsh_kwargs：传递给 Gmsh 网格生成器的参数
                min_sizing_field = algorithm.gmsh_kwargs.get("min_sizing_field")

                # 增加最大网格元素数字限制; 将最大网格元素数设为 100 万;
                # 确保在生成更细网格时不会因元素数限制而失败
                # 必要性：阻尼值较小时会生成更细密的网格，需要更多元素
                algorithm.max_mesh_elements = 1e6  # allow for a bunch more elements

                # 阻尼参数循环扫描, 遍历所有阻尼值
                # np.geomspace：生成几何对数空间的阻尼值序列
                # 示例：从 0.5 到 2.0，生成 20 个等比数列值
                for last_step_damping in np.geomspace(
                    start=last_step_damping_config.start, stop=last_step_damping_config.stop, num=last_step_damping_config.num_steps
                ):
                    # 更新算法中尺寸场阻尼对象的 last_step_damping 属性
                    # sizing_field_damping：尺寸场阻尼对象
                    # last_step_damping：最后一步的阻尼系数
                    # 控制尺寸场计算过程中的阻尼效果
                    algorithm.sizing_field_damping.last_step_damping = float(last_step_damping)
                    # 这个参数控制着尺寸场计算过程中的数值稳定性
                    # 影响网格生成算法在计算节点尺寸分布时的迭代行为

                    # 调整传递给 Gmsh 网格生成器的最小尺寸场参数
                    # 公式：新最小尺寸 = 原始最小尺寸 × 阻尼系数
                    # 物理意义：直接控制网格生成的精细程度（控制生成网格的绝对精细程度）
                    algorithm.gmsh_kwargs["min_sizing_field"] = min_sizing_field * last_step_damping # 通过乘法因子直接改变网格生成的物理约束
                    """
                        这里就是论文中在“分辨率缩放场”中提到的
                        “此外，在推断过程中，我们还可以设置 cT −1<1 以生成比专家网格分辨率更高的网格。”
                    """
                    test_logger.experiment.suffix = f"{last_step_damping:.2f}"   # 设置日志后缀, 在日志中区分不同阻尼值的运行结果
                    trainer.test(algorithm, dataloaders=dataloaders.get("test")) # 执行测试
            else:
                # No suffix, no adaptation to the algorithm needed
                # 无后缀，无需适配算法
                trainer.test(algorithm, dataloaders=dataloaders.get("test"))

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

# 辅助函数：获取检查点列表
# 从加载路径中获取所有检查点信息。
def get_ckpts(loading_path, checkpoint_loading: str) -> List[Dict[str, str]]:
    """
    Get the checkpoints to load from the loading path. The loading path should contain a folder for each job type, and
    each job type should contain a folder for each seed. Each seed folder should contain a checkpoints folder with the
    checkpoints to load. The checkpoints should be named according to the epoch they were saved at, or "last.ckpt".
    Args:
        loading_path:
        checkpoint_loading:

    Returns: A list of dictionaries, each containing the job type, seed, and checkpoint path of a given run.

    """
    ckpt_queue: List[Dict[str, str]] = []  # list of runs to execute one after the other.
    for job_type in os.listdir(loading_path):
        job_type_path = os.path.join(loading_path, job_type)
        for seed in os.listdir(job_type_path):
            seed_path = os.path.join(job_type_path, seed)
            checkpoint_path = os.path.join(seed_path, "checkpoints")
            if checkpoint_loading == "last":
                checkpoint_path = os.path.join(checkpoint_path, "last.ckpt")
            else:
                try:
                    checkpoint_epoch = int(checkpoint_loading)
                    checkpoint_path = os.path.join(checkpoint_path, f"checkpoint-epoch={checkpoint_epoch:02d}.ckpt")
                except:
                    raise ValueError(f"Invalid checkpoint: {checkpoint_loading}")

            ckpt_queue.append(
                {
                    "job_type": job_type,
                    "seed": seed,
                    "checkpoint_path": checkpoint_path,
                }
            )
    return ckpt_queue

# 主程序入口
if __name__ == "__main__":
    evaluation()

# 代码功能总结
# 配置加载：使用 Hydra 管理配置文件。
# 环境初始化：设置随机种子、日志、警告等。
# 数据准备：加载数据集和数据加载器。
# 模型加载：根据检查点路径加载训练好的模型。
# 评估执行：使用 Lightning Trainer 进行模型评估。
# 参数扫描：可选地对“最后一步阻尼”参数进行几何对数空间扫描。
# 结果记录：使用 EvaluationLogger 记录评估结果。
#
# 该脚本是一个批量评估脚本，支持：
# 多任务类型
# 多随机种子
# 多检查点
# 参数扫描评估

# 为什么在评估中使用Trainer？
# 1、统一接口和基础设施
    # trainer.test(algorithm, dataloaders=dataloaders.get("test"))
    # 一致性：无论训练还是评估，都使用相同的 Trainer.test() 方法
    # 标准化：确保评估过程与训练使用相同的设置（如精度、设备）
# 2、自动处理设备转移
# 3、分布式评估支持
# 4、自动精度转换
# 5、内置指标收集和日志

'''
开始阻尼扫描
    ↓
对于每个阻尼值 D：
    │
    ├─ 1. 更新尺寸场阻尼参数
    │      → 影响尺寸场计算的稳定性
    │
    ├─ 2. 缩放最小网格尺寸
    │      → 新尺寸 = 原尺寸 × D
    │
    ├─ 3. 设置日志标识
    │      → 后缀 = f"{D:.2f}"
    │
    └─ 4. 执行完整测试
          → 生成网格 → 求解 → 记录结果
    ↓
下一个阻尼值
'''

'''
问题：网格质量受两个因素影响
    ↓
因素1：尺寸场计算的准确性（稳定性）
    ↓ 控制方法：sizing_field_damping
    ↓ 作用：确保尺寸场平滑、合理
    
因素2：绝对网格尺寸限制  
    ↓ 控制方法：min_sizing_field × damping
    ↓ 作用：直接控制网格精细度
    
双重控制 → 更全面的网格质量调控
'''
# 阻尼的具体作用
# 1) sizing_field_damping：控制尺寸场迭代的稳定性（数值行为）
    # 尺寸场（Sizing Field）是定义网格疏密分布的标量场
    # 阻尼控制尺寸场迭代计算的稳定性
    # 小阻尼（<1.0）：保守更新，更稳定但收敛慢
    # 大阻尼（>1.0）：激进更新，可能更快但可能振荡
# 2) min_sizing_field × damping：控制生成网格的绝对精细度（物理限制）
    # 阻尼系数 实际最小尺寸	网格效果
    #  0.5	  基准×0.5	更细密的网格
    #  1.0	  基准×1.0	标准网格
    #  2.0	  基准×2.0	更粗糙的网格