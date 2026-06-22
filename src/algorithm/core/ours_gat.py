import warnings
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Batch

from src.algorithm.architecture import get_gnn
from src.algorithm.core.inference_step_output import InferenceStepOutput
from src.algorithm.core.mesh_generation_algorithm import MeshGenerationAlgorithm
from src.algorithm.dataloader.amber_data import AmberData
from src.algorithm.dataloader.amber_dataset import AmberDataset
from src.algorithm.loss.amber_loss import AmberLoss
from src.algorithm.loss.mesh_generation_loss import MeshGenerationLoss
from src.algorithm.visualization.amber_visualization import get_learner_plot
from src.helpers.custom_types import MeshGenerationStatus, MetricDict, PlotDict
from src.helpers.qol import aggregate_metrics, prefix_keys
from src.helpers.torch_util import detach, make_batch
from src.mesh_util.mesh_metrics import MeshMetrics
from src.tasks.domains.mesh_wrapper import MeshWrapper


def get_evaluation_step_name(inference_steps: int, current_step: int) -> str:
    return "last" if current_step == inference_steps - 1 else f"step{current_step}"


class OursGat(MeshGenerationAlgorithm):
    def __init__(self, algorithm_config: DictConfig, train_dataset: AmberDataset):
        self.inference_steps: int = algorithm_config.inference_steps
        self.sizing_field_damping: DictConfig = algorithm_config.sizing_field_damping
        self.loss_type = algorithm_config.get("loss_type", "mse")
        super().__init__(algorithm_config=algorithm_config, train_dataset=train_dataset)

    ###################
    # Algorithm setup #
    ###################

    def _get_optimization_criterion(self) -> MeshGenerationLoss:
        return AmberLoss(label_transform=self.prediction_transform, loss_type=self.loss_type)

    def _get_model(self) -> nn.Module:
        return get_gnn(
            architecture_config=self._get_architecture_config_with_physics_codex(),
            example_graph=self.train_dataset.first.observation,
        )

    #################
    # Training loop #
    #################

    def _training_step(self, batch: Batch, batch_idx: int) -> Tuple[torch.Tensor, MetricDict]:
        prediction_bundle = self._predict(batch, is_train=True, return_details=True)
        loss, differences = self.criterion(predictions=prediction_bundle, labels=batch.y, graph_batch=batch)
        with torch.no_grad():
            nodes_per_batch = batch.ptr[1:] - batch.ptr[:-1]
            nodes_per_batch = detach(nodes_per_batch)
            train_scalars = {
                "loss": loss.item(),
                "min_dif": torch.min(differences).item(),
                "max_dif": torch.max(differences).item(),
                "mean_dif": torch.sum(differences).item(),
                "batch_min_nodes": np.min(nodes_per_batch),
                "batch_max_nodes": np.max(nodes_per_batch),
                "batch_edges": batch.num_edges,
                "batch_nodes": batch.num_nodes,
                "batch_components": batch.num_edges + batch.num_nodes,
                "batch_graphs": batch.ptr.shape[0] - 1,
            }

        return loss, train_scalars

    def on_train_epoch_end(self) -> None:
        if self.inference_steps > 1:
            new_data_metrics = self._add_online_data()
            new_data_metrics = prefix_keys(new_data_metrics, "online", separator="/")
            self.log_dict(new_data_metrics, on_epoch=True, prog_bar=True)

        super().on_train_epoch_end()

    ##########################
    # Evaluation and testing #
    ##########################
    def _evaluate_data_point(self, data: AmberData) -> MetricDict:
        cumulative_elements = data.mesh.nelements
        evaluation_metrics = {}
        mesh_metrics = None
        for step in range(self.inference_steps):
            inference_output = self._inference_step(data)
            new_mesh = inference_output.output_mesh
            predictions = inference_output.predictions
            prediction_bundle = inference_output.prediction_bundle

            cumulative_elements += new_mesh.nelements
            differences = self.criterion.get_differences(
                predictions=predictions,
                labels=data.observation.y.to(self.device),
            )
            prediction_metrics = {
                "sf_MAE": torch.mean(differences).item(),
                "sf_MSE": torch.mean(differences**2).item(),
                "refinement_success": int(inference_output.refinement_success),
                "refinement_scaled": int(inference_output.refinement_scaled),
                "refinement_okay": int(inference_output.refinement_okay),
                "cum_elements": cumulative_elements,
                "cur_elements": new_mesh.nelements,
                "cum_element_ratio": cumulative_elements / data.expert_mesh.nelements,
                "cur_element_ratio": new_mesh.nelements / data.expert_mesh.nelements,
                "min_sf": torch.min(predictions).item(),
                "graph_size": data.observation.num_nodes + data.observation.num_edges,
            }
            if prediction_bundle is not None and hasattr(self.criterion, "get_prediction_diagnostics"):
                graph_batch = make_batch(data.observation).to(self.device)
                prediction_metrics |= self.criterion.get_prediction_diagnostics(
                    predictions=prediction_bundle,
                    labels=data.observation.y.to(self.device),
                    graph_batch=graph_batch,
                )

            if inference_output.refinement_okay:
                mesh_metrics = MeshMetrics(
                    metric_config=self.config.mesh_metrics,
                    reference_mesh=data.expert_mesh,
                    evaluated_mesh=new_mesh,
                    fem_problem=data.fem_problem,
                )()
                metric_dict = prediction_metrics | mesh_metrics
                step_name = get_evaluation_step_name(self.inference_steps, step)
                evaluation_metrics |= prefix_keys(metric_dict, prefix=step_name)
                data = AmberData.from_reference(reference=data, new_mesh=new_mesh)
            else:
                if mesh_metrics is None:
                    mesh_metrics = MeshMetrics(
                        metric_config=self.config.mesh_metrics,
                        reference_mesh=data.expert_mesh,
                        evaluated_mesh=new_mesh,
                        fem_problem=data.fem_problem,
                    )()
                for remaining_step in range(step, self.inference_steps):
                    remaining_dict = prediction_metrics | mesh_metrics
                    remaining_step_name = get_evaluation_step_name(self.inference_steps, remaining_step)
                    remaining_dict = prefix_keys(remaining_dict, prefix=remaining_step_name)
                    evaluation_metrics |= remaining_dict
                break
        return evaluation_metrics

    def _visualize_data_point(self, data: AmberData) -> PlotDict:
        plots = {}
        mesh_generation_status: MeshGenerationStatus = "success"
        for step in range(self.inference_steps):
            inference_output = self._inference_step(data)
            mesh_generation_status = inference_output.mesh_generation_status
            new_mesh = inference_output.output_mesh
            predictions: np.ndarray = detach(inference_output.predictions)
            if inference_output.refinement_okay:
                current_plot = get_learner_plot(
                    predicted_mesh=data.mesh,
                    labels=detach(data.observation.y),
                    predicted_sizing_field=predictions,
                    mesh_generation_status=mesh_generation_status,
                )
                plots[f"step{step}"] = current_plot
                data = AmberData.from_reference(reference=data, new_mesh=new_mesh)
            else:
                break

        fem_problem = data.fem_problem
        solution = fem_problem.calculate_solution(data.mesh) if fem_problem is not None else None
        final_plot = get_learner_plot(
            predicted_mesh=data.mesh,
            solution=solution,
            mesh_generation_status=mesh_generation_status,
        )
        plots["final"] = final_plot
        return plots

    def _clip_detach_dampen(self, sizing_field: torch.Tensor, refinement_depth: int, is_train: bool) -> np.ndarray:
        sizing_field = self._clip_detach(sizing_field)

        damping_factor = self.sizing_field_damping.damping_factor
        if damping_factor is None or damping_factor == 0:
            scaling = 1.0
        else:
            exponent = self.inference_steps - refinement_depth - 1
            scaling = 1 / (damping_factor**exponent)
        sizing_field = sizing_field * scaling

        if not is_train and self.inference_steps == refinement_depth + 1:
            last_step_damping = self.sizing_field_damping.get("last_step_damping", None)
            if last_step_damping is not None and last_step_damping != 1.0:
                sizing_field = sizing_field * last_step_damping

        return sizing_field

    def _inference_step(self, data: AmberData) -> InferenceStepOutput:
        graph = data.observation
        mesh = data.mesh
        mesh_generation_status: MeshGenerationStatus = "success"

        with torch.no_grad():
            prediction_bundle = self._predict(make_batch(graph), is_train=False, return_details=True)
            predictions = prediction_bundle["final"]
        sizing_field = self._clip_detach_dampen(predictions, data.refinement_depth, is_train=self.training)

        if self.max_mesh_elements is not None:
            approx_new_num_elements = self._estimate_new_num_elements(mesh, sizing_field)
            if approx_new_num_elements > self.max_mesh_elements:
                if self.force_mesh_generation:
                    from src.mesh_util.sizing_field_util import scale_sizing_field_to_budget

                    sizing_field = scale_sizing_field_to_budget(
                        sizing_field=sizing_field,
                        mesh=mesh,
                        max_elements=self.max_mesh_elements,
                        node_type=self.mesh_node_type,
                    )
                    mesh_generation_status = "scaled"
                else:
                    return InferenceStepOutput(
                        predictions,
                        mesh,
                        "failed",
                        prediction_bundle=prediction_bundle,
                    )

        try:
            from src.tasks.domains.update_mesh import update_mesh

            new_mesh = update_mesh(old_mesh=mesh.mesh, sizing_field=sizing_field, gmsh_kwargs=self.gmsh_kwargs)
        except Exception as e:
            warnings.warn(
                f"Mesh generation for sizing field of range {sizing_field.min()}, {sizing_field.max()} failed with error: {e}"
            )
            return InferenceStepOutput(
                predictions,
                mesh,
                "failed",
                prediction_bundle=prediction_bundle,
            )

        return InferenceStepOutput(
            predictions,
            new_mesh,
            mesh_generation_status=mesh_generation_status,
            prediction_bundle=prediction_bundle,
        )

    def _estimate_new_num_elements(self, mesh: MeshWrapper, sizing_field: np.ndarray) -> float:
        from src.mesh_util.sizing_field_util import sizing_field_to_num_elements

        return sizing_field_to_num_elements(mesh, sizing_field, node_type=self.mesh_node_type)

    def _add_online_data(self) -> MetricDict:
        metrics = []
        for _ in range(self.train_dataset.new_samples_per_epoch):
            data: AmberData = self.train_dataset.get_data_point()
            inference_output = self._inference_step(data)
            new_mesh = inference_output.output_mesh
            if inference_output.refinement_okay and (new_mesh.num_elements + new_mesh.num_vertices) < self.config.dataloader.batch_size:
                new_data = AmberData.from_reference(reference=data, new_mesh=new_mesh)
                if new_data.graph_size < self.config.dataloader.batch_size:
                    self.train_dataset.add_data(new_data)
                    self.normalizer.update_normalizers(new_data.observation)
                    batch_size_reached = 0
                else:
                    batch_size_reached = 1
            else:
                new_data = data
                batch_size_reached = 0

            metrics.append(
                {
                    "new_depth": new_data.refinement_depth,
                    "new_num_elements": new_data.mesh.nelements,
                    "refinement_success": int(inference_output.refinement_success),
                    "refinement_scaled": int(inference_output.refinement_scaled),
                    "refinement_okay": int(inference_output.refinement_okay),
                    "batch_size_reached": batch_size_reached,
                }
            )
        metrics = aggregate_metrics(metrics)
        metrics = metrics | self.train_dataset.get_metrics()
        return metrics
