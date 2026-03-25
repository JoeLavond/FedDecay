"""
trainer_decay.py — FedDecay trainer plug-in for FederatedScope
==============================================================

FedDecay modifies standard FedAvg by applying a within-round learning-rate
decay to each client's local update steps.  Instead of all K local steps
contributing equally to the aggregated model, later steps are down-weighted
by an increasing factor of beta (the decay parameter, 0 < beta <= 1).

Motivation
----------
In heterogeneous federated learning the later local gradient steps drift
further from the global optimum ("client drift").  Attenuating them makes
the aggregated update closer to the direction of the global gradient,
improving convergence on non-IID data.

Implementation strategy
-----------------------
FedDecay is implemented as a *wrapper* (a set of hooks) over an existing
FederatedScope trainer rather than as a standalone class.  This keeps the
change minimal and compatible with all existing trainer types
(cvtrainer, nlptrainer, graphminibatch_trainer, …).

The hook-based plug-in pattern works as follows:

1. Before each local update step (epoch or batch), save the current model
   weights as ``ctx.last_model``.
2. After the optimizer step, compute the weight difference
   Δθ = θ_current − θ_last.
3. Scale the difference by the decay factor appropriate for this step index k:
     - Exponential:  scale = beta^k          (default; 0 < beta <= 1)
     - Linear:       scale = max(0, 1 − beta·k)
4. Apply the scaled update: θ_new = θ_last + scale · Δθ

This is mathematically equivalent to multiplying the learning rate by the
same scale factor at local step k.  With beta = 1 (the default), scale = 1
at every step, so the behaviour is identical to vanilla FedAvg.

Step indexing:  k starts at 0 for the *first* local step within a round,
so the first update is always applied at full strength (beta^0 = 1).

Fine-tuning:  FedDecay should generally not be applied during the
personalisation fine-tuning stage, because the local model is being
adapted to a single client's distribution.  The finetune() override
temporarily replaces ctx.beta with ctx.finetune_beta (default 1.0) so
that fine-tuning steps are unattenuated.

Usage (via cfg):
    federate.method: decay
    trainer.beta: 0.6            # decay factor, e.g. 0.2 / 0.4 / 0.6 / 0.8
    trainer.decay_scheme: exponential   # or 'linear'
    trainer.model_on_batch_or_epoch: epoch   # granularity of decay
    trainer.finetune_beta: 1.0   # beta used during fine-tuning (no decay)
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


def wrap_decay(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    """Wrap an existing FederatedScope trainer with FedDecay hooks.

    This function mutates ``base_trainer`` in-place by registering a set of
    lifecycle hooks that implement within-round learning-rate decay.  The
    trainer's class is not changed; hooks are injected at specific trigger
    points (``on_fit_start``, ``on_epoch_start``, ``on_epoch_end``, etc.).

    Parameters
    ----------
    base_trainer : Type[GeneralTorchTrainer]
        Any FederatedScope trainer instance (e.g. CVTrainer, NLPTrainer).
        The trainer's ``cfg`` must already contain the decay settings
        populated by ``init_decay_ctx``.

    Returns
    -------
    Type[GeneralTorchTrainer]
        The same trainer instance, now augmented with decay hooks and the
        FedDecay-aware ``finetune`` method.
    """

    # --- Attribute-level setup -------------------------------------------
    # Reads cfg fields and stores them on the trainer context (ctx) so that
    # the hook functions below can access them without re-reading cfg each time.
    init_decay_ctx(base_trainer)
    model_on_batch_or_epoch = base_trainer.cfg.trainer.model_on_batch_or_epoch

    # Replace the standard finetune() method with the FedDecay-aware version
    # that temporarily freezes the decay factor to finetune_beta (default 1.0)
    # during the personalisation fine-tuning stage.
    base_trainer.finetune = types.MethodType(finetune, base_trainer)

    # --- Action-level hooks ----------------------------------------------

    # Hook 1: Reset the step counter and clear the stored model at the START
    # of every training round.  Also runs at the END of a round (cleanup).
    base_trainer.register_hook_in_train(
        new_hook=_hook_decay_init,
        trigger='on_fit_start',
        insert_pos=-1
    )

    # Hooks 2 & 3: Save the model weights before each local step and apply
    # the scaled update after each step.  The granularity (epoch vs. batch)
    # is controlled by cfg.trainer.model_on_batch_or_epoch.
    if model_on_batch_or_epoch == 'epoch':
        # Save current weights at the start of each local epoch …
        base_trainer.register_hook_in_train(
            new_hook=_hook_save_model,
            trigger='on_epoch_start',
            insert_pos=-1
        )
        # … and apply the decayed update at the end of that epoch.
        base_trainer.register_hook_in_train(
            new_hook=_hook_decay_model,
            trigger='on_epoch_end',
            insert_pos=-1
        )

    elif model_on_batch_or_epoch == 'batch':
        # Finer-grained variant: decay is applied after every mini-batch.
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

    # Hook 4: Increment the step index after every epoch.
    # This runs *after* _hook_decay_model so that the first epoch uses
    # step_iter = 0 (scale = beta^0 = 1, full strength).
    base_trainer.register_hook_in_train(
        new_hook=_hook_update_step,
        trigger='on_epoch_end',
        insert_pos=-1
    )

    # Hook 5: Re-run the init hook at the END of training to clean up
    # ctx.last_model (which holds a full model copy) and reset the counter.
    base_trainer.register_hook_in_train(
        new_hook=_hook_decay_init,
        trigger='on_fit_end',
        insert_pos=-1
    )

    return base_trainer


def init_decay_ctx(base_trainer):
    """Copy FedDecay hyper-parameters from cfg into the trainer context.

    Parameters
    ----------
    base_trainer : GeneralTorchTrainer
        Trainer whose ``ctx`` and ``cfg`` attributes are used.

    Notes
    -----
    This function is called once during ``wrap_decay`` setup.  Storing the
    values on ``ctx`` avoids repeated attribute lookups inside the hot-path
    hook functions.
    """
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    # Whether to apply decay once per local epoch or once per mini-batch.
    # 'epoch' is the standard setting used in the paper's experiments.
    ctx.model_on_batch_or_epoch = cfg.trainer.model_on_batch_or_epoch

    # The decay factor β ∈ (0, 1].  β = 1.0 recovers standard FedAvg.
    ctx.beta = cfg.trainer.beta

    # Decay factor used exclusively during the fine-tuning stage.
    # Defaults to 1.0 (no decay) so that fine-tuning steps are unattenuated.
    ctx.finetune_beta = cfg.trainer.finetune_beta

    # Which decay schedule to apply: 'exponential' (default) or 'linear'.
    ctx.decay_scheme = cfg.trainer.decay_scheme


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def _hook_decay_init(ctx):
    """Reset per-round state: clear the saved model and zero the step counter.

    Registered at both ``on_fit_start`` (beginning of a local training round)
    and ``on_fit_end`` (cleanup after the round).  Resetting at ``on_fit_end``
    also releases the memory held by ``ctx.last_model``.
    """
    ctx.last_model = []   # will be replaced by a full model copy each step
    ctx.step_iter = 0     # local step index k; resets to 0 each round


def _hook_save_model(ctx):
    """Take a snapshot of the current model weights before a local update step.

    The snapshot is stored in ``ctx.last_model`` and used by
    ``_hook_decay_model`` immediately after the optimiser step completes.

    Note: ``copy.deepcopy`` is required so that ``ctx.last_model`` is not
    aliased to ``ctx.model`` (which PyTorch updates in-place during training).
    """
    ctx.last_model = copy.deepcopy(ctx.model)


def _hook_update_step(ctx):
    """Increment the local step counter after each epoch/batch completes.

    ``ctx.step_iter`` is used as the exponent k in the decay scaling factor.
    It is incremented *after* ``_hook_decay_model`` so that the first local
    update step (k = 0) is always applied at full strength (β^0 = 1).
    """
    ctx.step_iter += 1


def _hook_decay_model(ctx):
    """Apply the FedDecay scaling to the model after each local update step.

    For each parameter tensor, the weight change produced by the optimiser is
    scaled by a factor that depends on the current step index k:

        Exponential (default):  scale = β^k
        Linear:                 scale = max(0, 1 − β·k)

    The model is then set to:
        θ_new = θ_last + scale · (θ_current − θ_last)

    With β = 1 (the default) the scaling is always 1 and the behaviour is
    identical to vanilla SGD / FedAvg.

    Parameters are updated in-place using ``torch.no_grad()`` to avoid
    creating an unintended computation graph node.
    """
    # Iterate over parameter pairs from the current model and the pre-step
    # snapshot simultaneously so that the tensors stay aligned.
    for current_parameter, last_parameter in zip(
            ctx.model.parameters(),        # post-SGD-step model weights
            ctx.last_model.parameters()    # pre-SGD-step snapshot weights
    ):
        # --- Compute the raw weight change produced by the optimiser ---
        current_weights = current_parameter.detach().clone()
        last_weights    = last_parameter.detach().clone()
        model_difference = current_weights - last_weights  # Δθ = θ_k − θ_{k-1}

        # --- Scale the weight change according to the chosen decay scheme ---
        if ctx.decay_scheme == 'exponential':
            # Exponential decay: scale = β^k
            # At step 0 (first local epoch): scale = β^0 = 1  (full update)
            # At step 1 (second local epoch): scale = β^1 = β  (attenuated)
            # At step k: scale = β^k  (increasingly attenuated for k > 0)
            scaling = ctx.beta ** ctx.step_iter
            scaled_model_difference = scaling * model_difference

        elif ctx.decay_scheme == 'linear':
            # Linear decay: scale = max(0, 1 − β·k)
            # The update goes to zero at step k = 1/β and stays at zero for
            # all subsequent steps.
            scaling = 1 - (ctx.beta * ctx.step_iter)
            scaling = max([scaling, 0])  # clamp to non-negative
            scaled_model_difference = scaling * model_difference

        else:
            # Unknown scheme: log a warning and fall back to no scaling (β=1).
            print(f'decay scheme {ctx.decay_scheme} not implemented')
            scaling = 1
            scaled_model_difference = scaling * model_difference

        # --- Write the scaled update back into the current model in-place ---
        # θ_new = θ_{k-1} + scale · Δθ
        with torch.no_grad():
            current_parameter.copy_(
                last_weights + scaled_model_difference
            )


# ---------------------------------------------------------------------------
# Fine-tuning override
# ---------------------------------------------------------------------------

def finetune(self, target_data_split_name="train", hooks_set=None):
    """FedDecay-aware fine-tuning.

    Identical to the standard FederatedScope ``finetune()`` implementation
    except that it temporarily replaces ``ctx.beta`` with ``ctx.finetune_beta``
    (default 1.0) for the duration of fine-tuning.

    This prevents the decay from attenuating fine-tuning steps, which are
    meant to adapt the global model as strongly as possible to the local
    client distribution.  The original beta value is restored once fine-tuning
    is complete.

    Parameters
    ----------
    target_data_split_name : str
        Which data split to fine-tune on (default: ``'train'``).
    hooks_set : optional
        Custom hook set to use during fine-tuning; passed through to
        ``self.train()``.
    """
    # --- Freeze parameter gradients if requested by cfg ---
    require_grad_changed_paras = set()
    if self.cfg.trainer.finetune.freeze_param != "":
        preserved_paras = self._param_filter(
            self.ctx.model.state_dict(),
            self.cfg.trainer.finetune.freeze_param)
        for name, param in self.ctx.model.named_parameters():
            if name not in preserved_paras and param.requires_grad is True:
                param.requires_grad = False
                require_grad_changed_paras.add(name)

    # --- Override the learning rate and step counts for fine-tuning ---
    original_lrs = []
    for g in self.ctx.optimizer.param_groups:
        original_lrs.append(g['lr'])
        g['lr'] = self.cfg.trainer.finetune.lr
    original_epoch_num           = self.ctx["num_train_epoch"]
    original_batch_num           = self.ctx["num_train_batch"]
    original_batch_num_last      = self.ctx["num_train_batch_last_epoch"]
    ft_num_train_batch = int(
        min(self.ctx["num_train_batch"], self.cfg.trainer.finetune.steps))
    self.ctx["num_train_epoch"] = int(
        max(1, self.cfg.trainer.finetune.steps / ft_num_train_batch))
    self.ctx["num_train_batch"] = ft_num_train_batch
    self.ctx["num_train_batch_last_epoch"] = int(
        self.cfg.trainer.finetune.steps % ft_num_train_batch)

    # --- Temporarily disable decay during fine-tuning ---
    # Save the training-phase beta and replace it with the finetune beta
    # (usually 1.0, meaning no decay) so local fine-tuning steps are
    # not attenuated.
    actual_beta      = self.ctx.beta
    self.ctx.beta    = self.ctx.finetune_beta

    # --- Run fine-tuning ---
    self.train(target_data_split_name, hooks_set)

    # --- Restore the training-phase decay factor ---
    self.ctx.beta = actual_beta

    # --- Restore gradient flags ---
    if len(require_grad_changed_paras) > 0:
        for name, param in self.ctx.model.named_parameters():
            if name in require_grad_changed_paras:
                param.requires_grad = True

    # --- Restore the original learning rate and step counts ---
    for i, g in enumerate(self.ctx.optimizer.param_groups):
        g['lr'] = original_lrs[i]

    self.ctx["num_train_epoch"]            = original_epoch_num
    self.ctx["num_train_batch"]            = original_batch_num
    self.ctx["num_train_batch_last_epoch"] = original_batch_num_last
