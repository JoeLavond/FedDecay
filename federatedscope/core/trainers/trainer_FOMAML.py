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


def wrap_FOMAML(

    base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:

    # ---------------- attribute-level plug-in -----------------------
    init_FOMAML_ctx(base_trainer)

    # ---------------- action-level plug-in -----------------------

    # initializations
    base_trainer.register_hook_in_train(
        new_hook=_hook_FOMAML_init,
        trigger='on_fit_start',
        insert_pos=-1
    )

    base_trainer.register_hook_in_train(
        new_hook=_hook_update_step,
        trigger='on_epoch_end',
        insert_pos=-1
    )

    base_trainer.register_hook_in_train(
        new_hook=_hook_append_model,
        trigger='on_epoch_end',
        insert_pos=-1
    )

    base_trainer.register_hook_in_train(
        new_hook=_hook_FOMAML_model,
        trigger='on_fit_end',
        insert_pos=-1
    )

    return base_trainer


def init_FOMAML_ctx(base_trainer):

    # initializations
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg
    ctx.local_update_steps = cfg.federate.local_update_steps


# initialize last model for storage
def _hook_FOMAML_init(ctx):
    ctx.models = [copy.deepcopy(ctx.model)]
    ctx.step_iter = 0


def _hook_update_step(ctx):
    ctx.step_iter += 1


# save current model as ONLY step
def _hook_append_model(ctx):

    if step_iter in (
        ctx.local_update_steps - 1, ctx.local_update_steps
    ):

        ctx.models.append(
            copy.deepcopy(ctx.model)
        )


def _hook_FOMAML_model(ctx):

    # iterate each parameter for all models simulataneously
    for (
        init_parameter, last_parameter, current_parameter
    ) in zip(*ctx.models):

        init_weights = init_parameter.detach().clone()

        # compute parameter change
        current_weights = current_parameter.detach().clone()
        last_weights = last_parameter.detach().clone()
        model_difference = current_weights - last_weights

        # add to the initialization
        with torch.no_grad():
            init_parameter.copy_(
                init_weights + model_difference
            )
