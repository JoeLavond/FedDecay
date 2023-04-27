import copy
import logging

import torch

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.optimizer import wrap_regularized_optimizer
# from federatedscope.core.trainers.utils import calculate_batch_epoch_num
from typing import Type

# ------
import types

# ------

logger = logging.getLogger(__name__)


def wrap_decay(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    # ---------------- attribute-level plug-in -----------------------
    init_decay_ctx(base_trainer)
    model_on_batch_or_epoch = base_trainer.cfg.trainer.model_on_batch_or_epoch

    # ------
    # modify finetune
    base_trainer.finetune = types.MethodType(finetune, base_trainer)
    # ------

    # ---------------- action-level plug-in -----------------------

    # initializations
    base_trainer.register_hook_in_train(
        new_hook=_hook_decay_init,
        trigger='on_fit_start',
        insert_pos=-1
    )

    # for exact decay at the end of every local update:
    # save model and decay
    if model_on_batch_or_epoch == 'epoch':

        base_trainer.register_hook_in_train(
            new_hook=_hook_save_model,
            trigger='on_epoch_start',
            insert_pos=-1
        )

        base_trainer.register_hook_in_train(
            new_hook=_hook_decay_model,
            trigger='on_epoch_end',
            insert_pos=-1
        )


    elif model_on_batch_or_epoch == 'batch':

        base_trainer.register_hook_in_train(
            new_hook=_hook_save_model,
            trigger='on_batch_start',
            insert_pos=-1
        )

        base_trainer.register_hook_in_train(
            new_hook=_hook_decay_model,
            trigger='on_batch_end',
            insert_pos=-1
        )

    # when to update decay coefficient
    base_trainer.register_hook_in_train(
        new_hook=_hook_update_step,
        trigger='on_epoch_end',
        insert_pos=-1
    )

    # cleanup
    base_trainer.register_hook_in_train(
        new_hook=_hook_decay_init,
        trigger='on_fit_end',
        insert_pos=-1
    )

    return base_trainer


def init_decay_ctx(base_trainer):
    # initializations
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    # decay frequency
    ctx.model_on_batch_or_epoch = cfg.trainer.model_on_batch_or_epoch

    # decay values
    ctx.beta = cfg.trainer.beta
    ctx.finetune_beta = cfg.trainer.finetune_beta
    ctx.decay_scheme = cfg.trainer.decay_scheme


# initialize last model for storage
def _hook_decay_init(ctx):
    ctx.last_model = []
    ctx.step_iter = 0


# save current model as ONLY step
def _hook_save_model(ctx):
    ctx.last_model = copy.deepcopy(ctx.model)


def _hook_update_step(ctx):
    ctx.step_iter += 1


def _hook_decay_model(ctx):
    # iterate each parameter for all models simulataneously
    for current_parameter, last_parameter in zip(
            ctx.model.parameters(),  # target model for updates
            ctx.last_model.parameters()  # last model values
    ):
        # compute parameter change between each successive model step
        current_weights = current_parameter.detach().clone()
        last_weights = last_parameter.detach().clone()
        model_difference = current_weights - last_weights

        # scale differences with exponential decay
        if ctx.decay_scheme == 'exponential':
            scaling = ctx.beta ** ctx.step_iter  # factor decay each step
            scaled_model_difference = scaling * model_difference
        elif ctx.decay_scheme == 'linear':
            scaling = 1 - (ctx.beta * ctx.step_iter)  # linear decay each step
            scaling = max([scaling, 0])  # force scaling to be non-negative
            scaled_model_difference = scaling * model_difference
        else:
            print(f'decay scheme {ctx.decay_scheme} not implemented')
            scaling = 1
            scaled_model_difference = scaling * model_difference

        # combine decayed steps and add to the initialization
        with torch.no_grad():
            current_parameter.copy_(
                last_weights + scaled_model_difference
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
