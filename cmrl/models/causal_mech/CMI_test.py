from typing import Optional, List, Dict, Union, MutableMapping
import pathlib
from functools import partial
from itertools import count

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
from stable_baselines3.common.logger import Logger

from cmrl.utils.variables import Variable
from cmrl.models.causal_mech.neural_causal_mech import NeuralCausalMech
from cmrl.models.causal_mech.util import variable_loss_func, train_func, eval_func


class CMITest(NeuralCausalMech):
    def __init__(
        self,
        name: str,
        input_variables: List[Variable],
        output_variables: List[Variable],
        # ensemble
        ensemble_num: int = 7,
        elite_num: int = 5,
        # cfgs
        network_cfg: Optional[DictConfig] = None,
        encoder_cfg: Optional[DictConfig] = None,
        decoder_cfg: Optional[DictConfig] = None,
        optimizer_cfg: Optional[DictConfig] = None,
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

        self.total_CMI_epoch = 0

        super(CMITest, self).__init__(
            name=name,
            input_variables=input_variables,
            output_variables=output_variables,
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
        self.network = instantiate(self.network_cfg)(
            input_dim=self.encoder_output_dim,
            output_dim=self.decoder_input_dim,
            extra_dims=[self.output_var_num, self.ensemble_num],
        ).to(self.device)

    def build_graph(self):
        self.graph = None

    def single_step_forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, _ = self.get_inputs_batch_size(inputs)

        inputs_tensor = torch.zeros(self.ensemble_num, batch_size, self.input_var_num, self.encoder_output_dim).to(self.device)
        for i, var in enumerate(self.input_variables):
            out = self.variable_encoders[var.name](inputs[var.name].to(self.device))
            inputs_tensor[:, :, i] = out

        reduced_inputs_tensor = self.reduce_encoder_output(inputs_tensor)
        output_tensor = self.network(reduced_inputs_tensor)

        outputs = {}
        for i, var in enumerate(self.output_variables):
            hid = output_tensor[i]
            outputs[var.name] = self.variable_decoders[var.name](hid)

        if self.residual:
            outputs = self.residual_outputs(inputs, outputs)
        return outputs

    @property
    def CMI_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.input_var_num + 1, self.output_var_num, self.input_var_num, dtype=torch.long)
        for i in range(self.input_var_num + 1):
            m = torch.ones(self.output_var_num, self.input_var_num)
            if i != self.input_var_num:
                m[:, i] = 0
            mask[i] = m
        return mask.to(self.device)

    def CMI_single_step_forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """when first step, inputs should be dict of str and Tensor with (ensemble-num, batch-size, specific-dim) shape,
        since twice step, the shape of Tensor becomes (input-var-num + 1, ensemble-num, batch-size, specific-dim)

        Args:
            inputs:

        Returns:

        """
        batch_size, extra_dim = self.get_inputs_batch_size(inputs)

        inputs_tensor = torch.empty(*extra_dim, self.ensemble_num, batch_size, self.input_var_num, self.encoder_output_dim).to(
            self.device
        )
        for i, var in enumerate(self.input_variables):
            out = self.variable_encoders[var.name](inputs[var.name].to(self.device))
            inputs_tensor[..., i, :] = out

        if len(extra_dim) == 0:
            # [..., output-var-num, input-var-num]
            mask = self.CMI_mask
            # [..., output-var-num, ensemble-num, batch-size, input-var-num]
            mask = mask.unsqueeze(-2).unsqueeze(-2)
            mask = mask.repeat((1,) * len(mask.shape[:-3]) + (self.ensemble_num, batch_size, 1))
            reduced_inputs_tensor = self.reduce_encoder_output(inputs_tensor, mask)
            assert (
                not torch.isinf(reduced_inputs_tensor).any() and not torch.isnan(reduced_inputs_tensor).any()
            ), "tensor must not be inf or nan"
            output_tensor = self.network(reduced_inputs_tensor)
        else:
            output_tensor = torch.empty(
                *extra_dim, self.output_var_num, self.ensemble_num, batch_size, self.decoder_input_dim
            ).to(self.device)

            CMI_mask = self.CMI_mask
            for i in range(self.input_var_num + 1):
                # [..., output-var-num, input-var-num]
                mask = CMI_mask[i]
                # [..., output-var-num, ensemble-num, batch-size, input-var-num]
                mask = mask.unsqueeze(-2).unsqueeze(-2)
                mask = mask.repeat((1,) * len(mask.shape[:-3]) + (self.ensemble_num, batch_size, 1))
                if i == len(inputs_tensor) - 1:
                    reduced_inputs_tensor = self.reduce_encoder_output(inputs_tensor[i], mask)
                    outs = self.network(reduced_inputs_tensor)
                    output_tensor[i] = outs
                else:
                    for j in range(self.output_var_num):
                        ins = inputs_tensor[-1]
                        ins[:, :, j] = inputs_tensor[i, :, :, j, :]
                        reduced_inputs_tensor = self.reduce_encoder_output(inputs_tensor[i], mask)
                        outs = self.network(reduced_inputs_tensor)
                        output_tensor[i, j] = outs[j]

        outputs = {}
        for i, var in enumerate(self.output_variables):
            hid = output_tensor[:, i]
            outputs[var.name] = self.variable_decoders[var.name](hid)

        if self.residual:
            outputs = self.residual_outputs(inputs, outputs)
        return outputs

    def CMI_forward(self, inputs: MutableMapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """inputs should be dict of str and Tensor with (ensemble-num, batch-size, specific-dim) shape

        Args:
            inputs:

        Returns:

        """
        if self.multi_step.startswith("forward-euler"):
            step_num = int(self.multi_step.split()[-1])

            outputs = {}
            for step in range(step_num):
                outputs = self.CMI_single_step_forward(inputs)
                # outputs shape: (input-var-num + 1, ensemble-num, batch-size, specific-dim * 2)
                # new inputs shape: (input-var-num + 1, ensemble-num, batch-size, specific-dim)
                if step == 0:
                    for name in filter(lambda s: s.startswith("act"), inputs.keys()):
                        inputs[name] = inputs[name][None, ...].repeat([self.input_var_num + 1, 1, 1, 1])
                if step < step_num - 1:
                    for name in filter(lambda s: s.startswith("obs"), inputs.keys()):
                        inputs[name] = outputs["next_{}".format(name)][..., : inputs[name].shape[-1]]
        else:
            raise NotImplementedError("multi-step method {} is not supported".format(self.multi_step))

        return outputs

    def calculate_CMI(self, nll_loss: torch.Tensor):
        nll_loss_diff = nll_loss[:-1] - nll_loss[-1]
        self.forward_mask = (nll_loss_diff.mean(dim=(1, 2)) > 1).to(torch.long)

        print(self.forward_mask)

    def learn(
        self,
        # loader
        train_loader: DataLoader,
        valid_loader: DataLoader,
        # model learning
        longest_epoch: int = -1,
        improvement_threshold: float = 0.01,
        patience: int = 5,
        work_dir: Optional[Union[str, pathlib.Path]] = None,
        **kwargs
    ):
        if self.discovery:
            epoch_iter = range(longest_epoch) if longest_epoch >= 0 else count()
            epochs_since_update = 0

            loss_func = partial(variable_loss_func, output_variables=self.output_variables, device=self.device)
            train = partial(train_func, forward=self.CMI_forward, optimizer=self.optimizer, loss_func=loss_func)
            eval = partial(eval_func, forward=self.CMI_forward, loss_func=loss_func)

            best_eval_loss = eval(valid_loader).mean(dim=(0, 2, 3))

            for epoch in epoch_iter:
                train_loss = train(train_loader)
                eval_loss = eval(valid_loader)

                improvement = (best_eval_loss - eval_loss.mean(dim=(0, 2, 3))) / torch.abs(best_eval_loss)
                if (improvement > improvement_threshold).any().item():
                    best_eval_loss = torch.minimum(best_eval_loss, eval_loss.mean(dim=(0, 2, 3)))
                    epochs_since_update = 0

                    self.calculate_CMI(eval_loss)
                else:
                    epochs_since_update += 1

                # log
                self.total_CMI_epoch += 1
                if self.logger is not None:
                    self.logger.record("{}-CMI-test/epoch".format(self.name), epoch)
                    self.logger.record("{}-CMI-test/epochs_since_update".format(self.name), epochs_since_update)
                    self.logger.record("{}-CMI-test/train_dataset_size".format(self.name), len(train_loader.dataset))
                    self.logger.record("{}-CMI-test/valid_dataset_size".format(self.name), len(valid_loader.dataset))
                    self.logger.record("{}-CMI-test/train_loss".format(self.name), train_loss.mean().item())
                    self.logger.record("{}-CMI-test/val_loss".format(self.name), eval_loss.mean().item())
                    self.logger.record("{}-CMI-test/best_val_loss".format(self.name), best_eval_loss.mean().item())

                    self.logger.dump(self.total_CMI_epoch)

                if patience and epochs_since_update >= patience:
                    break

        super(CMITest, self).learn(
            train_loader=train_loader,
            valid_loader=valid_loader,
            # model learning
            longest_epoch=longest_epoch,
            improvement_threshold=improvement_threshold,
            patience=patience,
            work_dir=work_dir,
            **kwargs
        )
