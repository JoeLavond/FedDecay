import copy
import logging

import torch

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.optimizer import wrap_regularized_optimizer
#from federatedscope.core.trainers.utils import calculate_batch_epoch_num
from typing import Type

# ------
import types
# ------

logger = logging.getLogger(__name__)


def wrap_exact_decay(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:

    # ---------------- attribute-level plug-in -----------------------
    init_decay_ctx(base_trainer)

    #assert base_trainer.cfg.federate.batch_or_epoch == 'epoch'
    #batch_or_epoch = base_trainer.cfg.federate.batch_or_epoch
    batch_or_epoch = base_trainer.cfg.trainer.batch_or_epoch

    # ------
    # modify finetune
    base_trainer.finetune = types.MethodType(finetune, base_trainer)
    # ------

    # ---------------- action-level plug-in -----------------------

    # initailize model steps at fit start
    base_trainer.register_hook_in_train(
        new_hook=_hook_on_fit_start_save_model,
        trigger='on_fit_start',
        insert_pos=-1
    )

    # save models at end of batch or epoch
    if batch_or_epoch == 'epoch':

        base_trainer.register_hook_in_train(
            new_hook=_hook_on_end_save_model,
            trigger='on_epoch_end',
            insert_pos=-1
        )

    elif batch_or_epoch == 'batch':

        base_trainer.register_hook_in_train(
            new_hook=_hook_on_end_save_model,
            trigger='on_batch_end',
            insert_pos=-1
        )

    """
    At fit end:
    2. Compute differences between model steps
    3. Replace model with decayed model steps
    """
    base_trainer.register_hook_in_train(
        new_hook=_hook_on_fit_end_decay_model,
        trigger='on_fit_end',
        insert_pos=-1
    )

    return base_trainer


def init_decay_ctx(base_trainer):

    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    ctx.global_model = ctx.model
    ctx.batch_or_epoch = cfg.trainer.batch_or_epoch
    ctx.beta = cfg.trainer.beta
    ctx.finetune_beta = cfg.trainer.finetune_beta


def _hook_on_fit_start_save_model(ctx):

    # save current model as ONLY step
    ctx.models = [ctx.global_model]

def _hook_on_end_save_model(ctx):

    # save current model as step
    ctx.models.append(ctx.global_model)


def _hook_on_fit_end_decay_model(ctx):

    print()
    print(f'current mode is {ctx.cur_mode}')
    print(f'current beta is {ctx.beta}')
    print(f'at fit end there exists {len(ctx.models)} iterations to decay')

    # iterate each parameter for all models simulataneously
    for parameter, *parameter_value_list in zip(
        ctx.global_model.parameters(),  # target model for updates
        *[model.parameters() for model in ctx.models]  # each model update
    ):

        # compute parameter change between each successive model step
        model_parameter_differences_list = [
            parameter_value_list[i + 1] - parameter_value_list[i]  # step difference
            for i in range(len(parameter_value_list) - 1)  # n - 1 differences for n steps
        ]

        # scale differences with exponential decay
        model_parameter_differences_list = [
            (ctx.beta ** i) * model_parameter_differences_list[i]
            for i in range(len(model_parameter_differences_list))
        ]

        # combine decayed steps and add to the initialization
        decayed_update = torch.stack(model_parameter_differences_list).sum(dim=0)
        with torch.no_grad():
            parameter.copy_(
                parameter_value_list[0] + decayed_update
            )


def finetune(self, target_data_split_name="train", hooks_set=None):

    # freeze the parameters during the fine-tune stage
    require_grad_changed_paras = set()
    if self.cfg.trainer.finetune.freeze_param != "":
        preserved_paras = self._param_filter(
            self.ctx.model.state_dict(),
            self.cfg.trainer.finetune.freeze_param)
        for name, param in self.ctx.model.named_parameters():
            if name not in preserved_paras and param.requires_grad is True:
                param.requires_grad = False
                require_grad_changed_paras.add(name)

    # change the optimization configs
    original_lrs = []
    for g in self.ctx.optimizer.param_groups:
        original_lrs.append(g['lr'])
        g['lr'] = self.cfg.trainer.finetune.lr
    original_epoch_num = self.ctx["num_train_epoch"]
    original_batch_num = self.ctx["num_train_batch"]
    original_batch_num_last = self.ctx["num_train_batch_last_epoch"]
    ft_num_train_batch = int(
        min(self.ctx["num_train_batch"], self.cfg.trainer.finetune.steps))
    self.ctx["num_train_epoch"] = int(
        max(1, self.cfg.trainer.finetune.steps / ft_num_train_batch))
    self.ctx["num_train_batch"] = ft_num_train_batch
    self.ctx["num_train_batch_last_epoch"] = int(
        self.cfg.trainer.finetune.steps % ft_num_train_batch)

    # -------------
    # freeze beta for finetuning
    actual_beta = self.ctx.beta
    self.ctx.beta = self.ctx.finetune_beta
    # -------------

    # do the fine-tuning process
    self.train(target_data_split_name, hooks_set)

    # -------------
    # restore beta after finetuning
    self.ctx.beta = actual_beta
    # -------------

    # restore the state before fine-tuning
    if len(require_grad_changed_paras) > 0:
        for name, param in self.ctx.model.named_parameters():
            if name in require_grad_changed_paras:
                param.requires_grad = True

    for i, g in enumerate(self.ctx.optimizer.param_groups):
        g['lr'] = original_lrs[i]

    self.ctx["num_train_epoch"] = original_epoch_num
    self.ctx["num_train_batch"] = original_batch_num
    self.ctx["num_train_batch_last_epoch"] = original_batch_num_last


