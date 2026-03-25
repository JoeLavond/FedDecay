"""
trainer_FOMAML.py — First-Order MAML trainer plug-in for FederatedScope
========================================================================

This module implements a First-Order Model-Agnostic Meta-Learning (FOMAML)
baseline for the FedDecay experimental comparisons.

Background
----------
MAML (Finn et al., 2017) trains a global model θ such that a small number of
gradient steps on any client's local data leads to a good personalised model.
The meta-gradient is:

    ∇_θ L = ∇_{θ_K} L       (exact MAML, requires second-order derivatives)

FOMAML (First-Order MAML) drops the second-order terms, approximating the
meta-gradient with only the last local gradient step:

    ∇_θ L ≈ ∇_{θ_K} L       (first-order approximation, no Hessian required)

In the federated setting, the effective global update contributed by each
client is:

    Δθ = θ_K − θ_{K−1}

That is, only the *last* local gradient step is used to update the global
model, rather than the sum of all K local steps as in FedAvg.

Implementation strategy
-----------------------
Like ``trainer_decay.py``, FOMAML is implemented as a hook-based wrapper over
an existing FederatedScope trainer.

Hooks:
  1. on_fit_start  → ``_hook_FOMAML_init``:  store a copy of the initial
                      model θ_0; reset the step counter.
  2. on_epoch_end  → ``_hook_update_step``:  increment the step counter.
  3. on_epoch_end  → ``_hook_append_model``: if this is the second-to-last or
                      last local epoch, save a copy of the model.
  4. on_fit_end    → ``_hook_FOMAML_model``: combine the saved snapshots to
                      compute the FOMAML update and write it back.

After all hooks run, ``ctx.models`` contains three snapshots:
    ctx.models[0] = θ_0         (initial model, from _hook_FOMAML_init)
    ctx.models[1] = θ_{K−1}     (second-to-last local step)
    ctx.models[2] = θ_K         (last local step)

The global contribution is then:
    θ_global ← θ_0 + (θ_K − θ_{K−1})

Usage (via cfg):
    federate.method: FOMAML
    federate.local_update_steps: K   # number of local update steps
"""

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
    """Wrap an existing FederatedScope trainer with FOMAML hooks.

    Registers the four lifecycle hooks that together implement the First-Order
    MAML update rule.  The trainer class is not changed; hooks are appended at
    ``on_fit_start``, ``on_epoch_end``, and ``on_fit_end``.

    Parameters
    ----------
    base_trainer : Type[GeneralTorchTrainer]
        Any FederatedScope trainer instance.  ``cfg.federate.local_update_steps``
        must be set to the intended number of local update epochs K.

    Returns
    -------
    Type[GeneralTorchTrainer]
        The same trainer instance, now augmented with FOMAML hooks.
    """

    # --- Attribute-level setup -------------------------------------------
    # Read cfg fields into ctx so hooks can access them cheaply.
    init_FOMAML_ctx(base_trainer)

    # --- Action-level hooks ----------------------------------------------

    # Hook 1: at the start of each local training round, save θ_0 and reset
    # the step counter and the list of model snapshots.
    base_trainer.register_hook_in_train(
        new_hook=_hook_FOMAML_init,
        trigger='on_fit_start',
        insert_pos=-1
    )

    # Hook 2: increment the step counter after each local epoch completes.
    # Runs before _hook_append_model so that step_iter reflects the epoch
    # that just finished when _hook_append_model reads it.
    base_trainer.register_hook_in_train(
        new_hook=_hook_update_step,
        trigger='on_epoch_end',
        insert_pos=-1
    )

    # Hook 3: conditionally save the model at the second-to-last and last
    # local epochs (steps K-1 and K).  These two snapshots are used by
    # _hook_FOMAML_model to compute the meta-gradient approximation.
    base_trainer.register_hook_in_train(
        new_hook=_hook_append_model,
        trigger='on_epoch_end',
        insert_pos=-1
    )

    # Hook 4: at the end of the local training round, compute the FOMAML
    # update and overwrite the model with the result.
    base_trainer.register_hook_in_train(
        new_hook=_hook_FOMAML_model,
        trigger='on_fit_end',
        insert_pos=-1
    )

    return base_trainer


def init_FOMAML_ctx(base_trainer):
    """Copy FOMAML hyper-parameters from cfg into the trainer context.

    Parameters
    ----------
    base_trainer : GeneralTorchTrainer
        Trainer whose ``ctx`` and ``cfg`` attributes are used.
    """
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    # Total number of local update steps (epochs) per communication round.
    # Used in _hook_append_model to identify the final two local epochs.
    ctx.local_update_steps = cfg.federate.local_update_steps


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def _hook_FOMAML_init(ctx):
    """Initialise per-round state for FOMAML at the start of each local round.

    Saves a deep copy of the current (global) model as θ_0 and resets
    ``ctx.step_iter`` to 0.  The initial snapshot is the first element of
    ``ctx.models``; subsequent snapshots are appended by ``_hook_append_model``.
    """
    # ctx.models will hold [θ_0, θ_{K-1}, θ_K] after all hooks complete.
    ctx.models = [copy.deepcopy(ctx.model)]  # θ_0: global model at round start
    ctx.step_iter = 0                         # local epoch counter (1-based after _hook_update_step)


def _hook_update_step(ctx):
    """Increment the local epoch counter after each training epoch."""
    ctx.step_iter += 1


def _hook_append_model(ctx):
    """Save the model at the second-to-last and last local training epochs.

    Only the final two snapshots are needed to compute the FOMAML update:
        Δθ_FOMAML = θ_K − θ_{K−1}

    This hook appends a deep copy of the model to ``ctx.models`` when the
    current step index (already incremented by ``_hook_update_step``) equals
    K−1 or K, where K = ``ctx.local_update_steps``.

    After both calls complete, ``ctx.models`` is:
        [θ_0, θ_{K-1}, θ_K]
    ready for ``_hook_FOMAML_model`` to consume.
    """
    # ctx.step_iter has already been incremented by _hook_update_step for
    # this epoch, so it equals the 1-based index of the epoch that just ran.
    if ctx.step_iter in (ctx.local_update_steps - 1, ctx.local_update_steps):
        # Append a frozen snapshot of the model at this local step.
        ctx.models.append(
            copy.deepcopy(ctx.model)
        )


def _hook_FOMAML_model(ctx):
    """Apply the FOMAML meta-update at the end of the local training round.

    After all local epochs complete, ``ctx.models`` contains three snapshots:
        ctx.models[0] = θ_0     (initial global model)
        ctx.models[1] = θ_{K-1} (second-to-last local step)
        ctx.models[2] = θ_K     (last local step)

    The FOMAML update sets the model to:
        θ_new = θ_0 + (θ_K − θ_{K-1})

    This uses only the last local gradient step as an approximation to the
    meta-gradient, without requiring second-order (Hessian) information.
    Parameters are updated in-place on ``ctx.models[0]``, which FederatedScope
    will read back as the trained model for this round.
    """
    # Iterate over parameter tensors from all three model snapshots in lockstep.
    # .parameters() yields tensors in a consistent, deterministic order for any
    # given nn.Module, so zipping the three generators keeps tensors aligned.
    for (
        init_parameter, last_parameter, current_parameter
    ) in zip(
        ctx.models[0].parameters(),  # θ_0: initial global model
        ctx.models[1].parameters(),  # θ_{K-1}: second-to-last local step
        ctx.models[2].parameters(),  # θ_K: last local step
    ):
        # θ_0: the global model weights at the start of this round
        init_weights = init_parameter.detach().clone()

        # Compute the last local gradient step: Δθ = θ_K − θ_{K-1}
        current_weights  = current_parameter.detach().clone()  # θ_K
        last_weights     = last_parameter.detach().clone()     # θ_{K-1}
        model_difference = current_weights - last_weights      # Δθ_FOMAML

        # Apply the FOMAML update: θ_new = θ_0 + Δθ_FOMAML
        # This replaces the full K-step local update with a single-step
        # meta-gradient approximation anchored at the initial global model.
        with torch.no_grad():
            init_parameter.copy_(
                init_weights + model_difference
            )
