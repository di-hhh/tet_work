from collections import deque
from typing import Dict, List

import numpy as np
from omegaconf import DictConfig

from src.algorithm.dataloader.amber_data import AmberData
from src.algorithm.dataloader.mesh_generation_dataset import MeshGenerationDataset


class AmberDataset(MeshGenerationDataset):
    def __init__(self, *, algorithm_config: DictConfig, persistent_data: List[AmberData]):
        """
        Args:
            algorithm_config: The configuration for the algorithm.
            persistent_data: List of persistent graph data (optional).
        """
        super().__init__(algorithm_config=algorithm_config, persistent_data=persistent_data)
        dataloader_config = algorithm_config.dataloader
        self.max_buffer_size = dataloader_config.max_size
        self.add_strategy = dataloader_config.add_strategy

        self.new_samples_per_epoch = dataloader_config.new_samples_per_epoch

        if self.new_samples_per_epoch == "auto":
            # 每训练周期添加到缓冲区的新样本数量
            steps_per_epoch = dataloader_config.steps_per_epoch
            self.new_samples_per_epoch = steps_per_epoch // 8

        self.max_mesh_depth = dataloader_config.max_mesh_depth
        if self.max_mesh_depth == "auto":
            # allow for a maximum depth (number of previous refinements) of a mesh equal to the number of
            # inference steps. In particular, this means that "final" meshes that have been fully refined are
            # allowed in the training data, even though they are never predicted on during inference.
            # This is an additional bit of data augmentation, allowing for smaller sizing fields during training
            # than should be necessary iid during inference.

            # 允许网格的最大深度（即先前细化的次数）等于推理步骤的数量。
            # 特别地，这意味着经过完全细化的"最终"网格也允许出现在训练数据中，即使它们在推理过程中从未被预测过。
            # 这是一种额外的数据增强手段，使得在训练期间可以采用比推理时独立同分布所需的更小尺寸场。

            self.max_mesh_depth = algorithm_config.inference_steps

        self.sizing_field_interpolation_type = algorithm_config.sizing_field_interpolation_type

        # 在线数据, 先进先出队列
        self._online_data = deque(maxlen=self.max_buffer_size - len(self._persistent_data))  # Online FIFO buffer

    def add_data(self, new_data: AmberData):
        """
        Add new graphs to the online buffer and get the index of the latest addition.
        将新图添加到在线缓冲区，并获取最新添加项的索引。
        """
        self._online_data.append(new_data)
        return len(self) - 1  # New data index

    def get_data_point(self) -> AmberData:
        """
        Get a random data point from the buffer to use for creating new online data.
        If a maximum depth
        (which is defined as the number of refinements/generation iterations starting from the initial mesh)
        is specified, only data points with a depth less than the maximum depth are considered, such that the refined
        mesh has the maximum depth.
        Returns: A random data point

        从缓冲区获取一个随机数据点，用于生成新的在线数据。
        若指定了最大深度（定义为从初始网格开始经过的细化次数/生成迭代次数），
        则仅考虑深度小于最大深度的数据点，
        以确保细化后的网格达到该最大深度。
        返回：一个随机数据点

        """

        if self.add_strategy == "stratified":
            # 如果设置为 "stratified"，先确定有效深度，然后以该深度进行采样
            assert (
                self.max_mesh_depth is not None and self.max_mesh_depth > 0
            ), f"Invalid self.max_mesh_depth {self.max_mesh_depth} for stratified sampling"
            # Stratified sampling: Sample a target depth first, then randomly take a sample that matches this depth.
            # If there are no samples with this depth (yet), take a random sample.

            # Efficient extraction of depths and filtering using list comprehension and set for uniqueness
            valid_depths = set(p.refinement_depth for p in self.data if p.refinement_depth < self.max_mesh_depth)
            if not valid_depths:
                raise ValueError(f"No valid data points with {self.max_mesh_depth=}")
            target_depth = np.random.choice(list(valid_depths))

            # Collect indices of all data points matching the target depth in one pass, select one of them randomly
            # 一次性收集所有符合目标深度的数据点索引，然后从中随机选择一个。
            selected_indices = [i for i, p in enumerate(self.data) if p.refinement_depth == target_depth]
            idx = np.random.choice(selected_indices)

        elif self.add_strategy == "random":
            # Random sampling: Sample a random valid data point by taking a random permutation of the data and
            # selecting the first valid data point. This point always exists because we have protected data with a
            # depth of 0.
            permutation = np.random.permutation(len(self))
            position = 0
            if self.max_mesh_depth is not None:
                assert self.max_mesh_depth > 0
                while self.data[permutation[position]].refinement_depth >= self.max_mesh_depth:
                    position += 1
            idx = permutation[position]

        else:
            raise ValueError(f"Unknown strategy {self.add_strategy}")
        return self.data[idx]

    @property
    def is_full(self) -> bool:
        """
        Checks if the buffer has reached its maximum size.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return len(self._online_data) == self._online_data.maxlen

    @property
    def data(self) -> List[AmberData]:
        """
        Returns the combined list of protected data and those currently in the buffer.

        Returns: A list of AMBERData objects
        """
        return self._persistent_data + list(self._online_data)

    def get_metrics(self) -> Dict[str, float]:
        metrics = super().get_metrics()
        largest_graph = np.max([data.graph_size for data in self.data])
        avg_sampled_count = np.mean([data.sampled_count for data in self.data])
        std_sampled_count = np.std([data.sampled_count for data in self.data])
        return metrics | {"largest_graph": largest_graph, "avg_sampled_count": avg_sampled_count, "std_sampled_count": std_sampled_count}
