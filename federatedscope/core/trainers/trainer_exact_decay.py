import copy
import logging

import torch

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.optimizer import wrap_regularized_optimizer
#from federatedscope.core.trainers.utils import calculate_batch_epoch_num
from typing import Type

logger = logging.getLogger(__name__)


def wrap_exact_decay(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:

    # ---------------- attribute-level plug-in -----------------------
    init_decay_ctx(base_trainer)

    #assert base_trainer.cfg.federate.batch_or_epoch == 'epoch'
    batch_or_epoch = base_trainer.cfg.federate.batch_or_epoch

    # ---------------- action-level plug-in -----------------------

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
    ctx.models = [ctx.model]

    ctx.beta = cfg.trainer.beta


def _hook_on_end_save_model(ctx):

    # save current model as step
    ctx.models.append(ctx.global_model)


def _hook_on_fit_end_decay_model(ctx):

    print(f'At fit end there exists {len(ctx.models)} iterations to decay')

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


