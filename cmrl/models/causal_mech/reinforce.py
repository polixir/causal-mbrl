from typing import List, Optional, Dict, Union, MutableMapping, Tuple
import math
import pathlib
from itertools import count
from functools import partial
import copy

import torch
import numpy as np
from torch.utils.data import DataLoader
from stable_baselines3.common.logger import Logger
from omegaconf import DictConfig
from hydra.utils import instantiate

from cmrl.utils.variables import Variable
from cmrl.models.causal_mech.neural_causal_mech import NeuralCausalMech
from cmrl.models.graphs.prob_graph import BernoulliGraph
from cmrl.models.causal_mech.util import variable_loss_func, train_func, eval_func

default_graph_optimizer_cfg = DictConfig(
    dict(
        _target_="torch.optim.Adam",
        _partial_=True,
        lr=1e-3,
        weight_decay=0.0,
        eps=1e-8,
    )
)


class ReinforceCausalMech(NeuralCausalMech):
    def __init__(
        self,
        name: str,
        input_variables: List[Variable],
        output_variables: List[Variable],
        # model learning
        longest_epoch: int = -1,
        improvement_threshold: float = 0.01,
        patience: int = 5,
        # ensemble
        ensemble_num: int = 7,
        elite_num: int = 5,
        # cfgs
        network_cfg: Optional[DictConfig] = None,
        encoder_cfg: Optional[DictConfig] = None,
        decoder_cfg: Optional[DictConfig] = None,
        optimizer_cfg: Optional[DictConfig] = None,
        graph_optimizer_cfg: Optional[DictConfig] = default_graph_optimizer_cfg,
        # graph params
        concat_mask: bool = True,
        graph_MC_samples: int = 100,
        graph_max_stack: int = 200,
        lambda_sparse: float = 1e-3,
        # forward method
        residual: bool = True,
        encoder_reduction: str = "sum",
        multi_step: str = "none",
        # logger
        logger: Optional[Logger] = None,
        # others
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        if multi_step == "none":
            multi_step = "forward-euler 1"

        # cfgs
        self.graph_optimizer_cfg = graph_optimizer_cfg

        # graph params
        self._concat_mask = concat_mask
        self._graph_MC_samples = graph_MC_samples
        self._graph_max_stack = graph_max_stack
        self._lambda_sparse = lambda_sparse

        self.graph_optimizer = None

        super(ReinforceCausalMech, self).__init__(
            name=name,
            input_variables=input_variables,
            output_variables=output_variables,
            longest_epoch=longest_epoch,
            improvement_threshold=improvement_threshold,
            patience=patience,
            ensemble_num=ensemble_num,
            elite_num=elite_num,
            network_cfg=network_cfg,
            encoder_cfg=encoder_cfg,
            decoder_cfg=decoder_cfg,
            optimizer_cfg=optimizer_cfg,
            residual=residual,
            encoder_reduction=encoder_reduction,
            multi_step=multi_step,
            logger=logger,
            device=device,
            **kwargs
        )

    def build_network(self):
        input_dim = self.encoder_output_dim
        if self._concat_mask:
            input_dim += self.input_var_num

        self.network = instantiate(self.network_cfg)(
            input_dim=input_dim,
            output_dim=self.decoder_input_dim,
            extra_dims=[self.output_var_num, self.ensemble_num],
        ).to(self.device)

    def build_graph(self):
        self.graph = BernoulliGraph(
            in_dim=self.input_var_num,
            out_dim=self.output_var_num,
            include_input=False,
            init_param=1e-6,
            requires_grad=True,
            device=self.device,
        )

    def build_optimizer(self):
        assert (
            self.network is not None and self.graph is not None
        ), "network and graph are both required when building optimizer"
        super().build_optimizer()

        # graph optimizer
        self.graph_optimizer = instantiate(self.graph_optimizer_cfg)(self.graph.parameters)

    @property
    def causal_graph(self) -> torch.Tensor:
        """property causal graph"""
        assert self.graph is not None, "graph incorrectly initialized"

        return self.graph.get_binary_adj_matrix(threshold=0.5)

    def single_step_forward(
        self,
        inputs: MutableMapping[str, torch.Tensor],
        train: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, extra_dim = self.get_inputs_batch_size(inputs)
        assert len(extra_dim) == 0, "unexpected dimension in the inputs"

        inputs_tensor = torch.zeros(self.ensemble_num, batch_size, self.input_var_num, self.encoder_output_dim).to(self.device)
        for i, var in enumerate(self.input_variables):
            out = self.variable_encoders[var.name](inputs[var.name].to(self.device))
            inputs_tensor[..., i, :] = out

        if train and self.discovery:
            # [ensemble-num, batch-size, input-var-num, output-var-num]
            adj_matrix = self.graph.sample(None, sample_size=(self.ensemble_num, batch_size))
            # [ensemble-num, batch-size, output-var-num, input-var-num]
            mask = adj_matrix.transpose(-1, -2)
            # [output-var-num, ensemble-num, batch-size, input-var-num]
            mask = mask.permute(2, 0, 1, 3)
        else:
            if mask is None:
                mask = self.forward_mask
                mask = mask.unsqueeze(-2).unsqueeze(-2)
                mask = mask.repeat(1, self.ensemble_num, batch_size, 1)

        # [output-var-num, ensemble-num, batch-size, encoder-output-dim]
        reduced_inputs_tensor = self.reduce_encoder_output(inputs_tensor, mask=mask)
        if self._concat_mask:
            # [output-var-num, ensemble-num, batch-size, encoder-output-dim + input-var-num]
            reduced_inputs_tensor = torch.cat([reduced_inputs_tensor, mask], dim=-1)
        output_tensor = self.network(reduced_inputs_tensor)

        outputs = {}
        for i, var in enumerate(self.output_variables):
            hid = output_tensor[i]
            outputs[var.name] = self.variable_decoders[var.name](hid)

        if self.residual:
            outputs = self.residual_outputs(inputs, outputs)
        return outputs

    def forward(
        self,
        inputs: MutableMapping[str, torch.Tensor],
        train: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.multi_step.startswith("forward-euler"):
            step_num = int(self.multi_step.split()[-1])

            outputs = {}
            for step in range(step_num):
                outputs = self.single_step_forward(inputs, train=train, mask=mask)
                if step < step_num - 1:
                    for name in filter(lambda s: s.startswith("obs"), inputs.keys()):
                        inputs[name] = outputs["next_{}".format(name)][..., : inputs[name].shape[-1]]
        else:
            raise NotImplementedError("multi-step method {} is not supported".format(self.multi_step))

        return outputs

    def train_graph(self, loader: DataLoader, data_ratio: float):
        num_batches = len(loader)
        train_num = int(num_batches * data_ratio)

        grads = torch.tensor([0], dtype=torch.float32)
        for i, (inputs, targets) in enumerate(loader):
            if train_num <= i:
                break

            grads = grads + self._update_graph(inputs, targets)

        return grads

    def _update_graph(
        self,
        inputs: MutableMapping[str, torch.Tensor],
        targets: MutableMapping[str, torch.Tensor],
    ) -> torch.Tensor:
        # do Monte-Carlo sampling to obtain adjacent matrices and corresponding model losses
        adj_matrices, losses = self._MC_sample(inputs, targets)

        # calculate graph gradients
        graph_grads = self._estimate_graph_grads(adj_matrices, losses)

        # update graph
        graph_params = self.graph.parameters[0]  # only one tensor parameter
        self.graph_optimizer.zero_grad()
        graph_params.grad = graph_grads
        self.graph_optimizer.step()

        return graph_grads.detach().cpu()

    def _MC_sample(
        self,
        inputs: MutableMapping[str, torch.Tensor],
        targets: MutableMapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        num_graph_list = [
            min(self._graph_max_stack, self._graph_MC_samples - i * self._graph_max_stack)
            for i in range(math.ceil(self._graph_MC_samples / self._graph_max_stack))
        ]
        num_graph_list = [(num_graph_list[i], sum(num_graph_list[:i])) for i in range(len(num_graph_list))]

        # sample graphs
        adj_mats = self.graph.sample(None, sample_size=self._graph_MC_samples)

        # evaluate scores using the sampled adjacency matrices and data
        batch_size, extra_dim = self.get_inputs_batch_size(inputs)
        assert len(extra_dim) == 0, "unexpected dimension in the inputs"

        losses = []
        for graph_count, start_idx in num_graph_list:
            # [ensemble-num, samples*batch_size, input-var-num, output-var-num]
            expanded_adj_mats = (
                adj_mats[None, start_idx : start_idx + graph_count, None]
                .expand(self.ensemble_num, -1, batch_size, -1, -1)
                .flatten(1, 2)
            )
            expanded_masks = expanded_adj_mats.transpose(-1, -2).permute(2, 0, 1, 3)

            expanded_inputs = {}
            expanded_targets = {}
            # expand inputs and targets
            for in_key in inputs:
                expanded_inputs[in_key] = inputs[in_key].repeat(1, graph_count, 1)
            for tar_key in targets:
                expanded_targets[tar_key] = targets[tar_key].repeat(1, graph_count, 1)

            with torch.no_grad():
                outputs = self.forward(expanded_inputs, train=False, mask=expanded_masks)
            loss = variable_loss_func(outputs, expanded_targets, self.output_variables, device=self.device)
            loss = loss.reshape(loss.shape[0], graph_count, batch_size, -1)
            losses.append(loss.mean(dim=(0, 2)))
        losses = sum(losses)

        return adj_mats, losses

    def _estimate_graph_grads(
        self,
        adj_matrices: torch.Tensor,
        losses: torch.Tensor,
    ) -> torch.Tensor:
        """Use MC samples and corresponding losses to estimate gradients via REINFORCE.

        Args:
            adj_matrices (tensor): MC sampled adjacent matrices from current graph,
                shaped [num-samples, input-var-num, output-var-num].
            losses (tensor): the model losses corresponding to the adjacent matrices,
                shaped [num-samples, output-var-num]

        """
        num_graphs = adj_matrices.shape[0]
        losses = losses.unsqueeze(dim=1)

        # calculate graph gradients
        edge_prob = self.graph.get_adj_matrix()
        num_pos = adj_matrices.sum(dim=0)
        num_neg = num_graphs - num_pos
        mask = ((num_pos > 0) * (num_neg > 0)).float()
        pos_grads = (losses * adj_matrices).sum(dim=0) / num_pos.clamp_(min=1e-5)
        neg_grads = (losses * (1 - adj_matrices)).sum(dim=0) / num_neg.clamp_(min=1e-5)
        graph_grads = mask * edge_prob * (1 - edge_prob) * (pos_grads - neg_grads + self._lambda_sparse)

        return graph_grads

    def learn(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        graph_data_ratio: float = 0.5,
        train_graph_freq: int = 2,
        work_dir: Optional[Union[str, pathlib.Path]] = None,
        **kwargs
    ):
        assert 0 <= graph_data_ratio <= 1, "graph data ratio should be in [0, 1]"

        best_weights: Optional[Dict] = None
        epoch_iter = range(self.longest_epoch) if self.longest_epoch >= 0 else count()
        epochs_since_update = 0

        loss_fn = partial(variable_loss_func, output_variables=self.output_variables, device=self.device)
        train_fn = partial(train_func, forward=partial(self.forward, train=True), optimizer=self.optimizer, loss_func=loss_fn)
        eval_fn = partial(eval_func, forward=partial(self.forward, train=False), loss_func=loss_fn)

        best_eval_loss = eval_fn(valid_loader).mean(dim=(-2, -1))
        for epoch in epoch_iter:
            if self.discovery and epoch % train_graph_freq == 0:
                grads = self.train_graph(train_loader, data_ratio=graph_data_ratio)
                print(self.graph.parameters[0])
                print(self.graph.get_binary_adj_matrix())

            train_loss = train_fn(train_loader)
            eval_loss = eval_fn(valid_loader)

            maybe_best_weights = self._maybe_get_best_weights(
                best_eval_loss, eval_loss.mean(dim=(-2, -1)), self.improvement_threshold
            )
            if maybe_best_weights:
                # best loss
                best_eval_loss = torch.minimum(best_eval_loss, eval_loss.mean(dim=(-2, -1)))
                best_weights = maybe_best_weights
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            # log
            self.total_epoch += 1
            if self.logger is not None:
                self.logger.record("{}/epoch".format(self.name), epoch)
                self.logger.record("{}/epoch_since_update".format(self.name), epochs_since_update)
                self.logger.record("{}/train_dataset_size".format(self.name), len(train_loader.dataset))
                self.logger.record("{}/valid_dataset_size".format(self.name), len(valid_loader.dataset))
                self.logger.record("{}/train_loss".format(self.name), train_loss.mean().item())
                self.logger.record("{}/val_loss".format(self.name), eval_loss.mean().item())
                self.logger.record("{}/best_val_loss".format(self.name), best_eval_loss.mean().item())

                if self.discovery and epoch % train_graph_freq == 0:
                    self.logger.record("{}/graph_update_grads".format(self.name), grads.abs().mean().item())

                self.logger.dump(self.total_epoch)

            if self.patience and epochs_since_update >= self.patience:
                break

        # saving the best models
        self._maybe_set_best_weights_and_elite(best_weights, best_eval_loss)

        self.save(save_dir=work_dir)

    def _maybe_get_best_weights(
        self, best_val_loss: torch.Tensor, val_loss: torch.Tensor, threshold: float = 0.01
    ) -> Optional[Dict]:
        improvement = (best_val_loss - val_loss) / torch.abs(best_val_loss)
        if (improvement > threshold).any().item():
            best_weights = {
                "graph": copy.deepcopy(self.graph.parameters[0].detach().clone()),
                "model": copy.deepcopy(self.network.state_dict()),
            }
        else:
            best_weights = None

        return best_weights

    def _maybe_set_best_weights_and_elite(self, best_weights: Optional[Dict], best_val_score: torch.Tensor):
        if best_weights is not None:
            self.network.load_state_dict(best_weights["model"])
            self.graph.set_data(best_weights["graph"])

        sorted_indices = np.argsort(best_val_score.tolist())
        self.elite_indices = sorted_indices[: self.elite_num]
