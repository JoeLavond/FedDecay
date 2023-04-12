from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_training_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # Trainer related options
    # ------------------------------------------------------------------------ #
    cfg.trainer = CN()
    cfg.trainer.type = 'general'

    # ------
    # when should model be decayed?
    # note that decay coefficients update every epoch
    cfg.trainer.model_on_batch_or_epoch = 'epoch'
    # default decay
    cfg.trainer.beta = 1.0
    cfg.trainer.finetune_beta = 1.0
    # ------

    cfg.trainer.finetune = CN()
    cfg.trainer.finetune.before_eval = False
    cfg.trainer.finetune.steps = 5
    cfg.trainer.finetune.lr = 0.01
    cfg.trainer.finetune.freeze_param = ""  # parameters frozen in fine-tuning stage
    # cfg.trainer.finetune.only_psn = True

    # ------------------------------------------------------------------------ #
    # Optimizer related options
    # ------------------------------------------------------------------------ #
    cfg.optimizer = CN()

    cfg.optimizer.type = 'SGD'
    cfg.optimizer.lr = 0.1
    cfg.optimizer.weight_decay = .0
    cfg.optimizer.momentum = .0
    cfg.optimizer.grad_clip = -1.0  # negative numbers indicate we do not clip grad

    # ------------------------------------------------------------------------ #
    # lr_scheduler related options
    # ------------------------------------------------------------------------ #
    # cfg.lr_scheduler = CN()
    # cfg.lr_scheduler.type = 'StepLR'
    # cfg.lr_scheduler.schlr_params = dict()

    # ------------------------------------------------------------------------ #
    # Early stopping related options
    # ------------------------------------------------------------------------ #
    cfg.early_stop = CN()

    # patience (int): How long to wait after last time the monitored metric improved.
    # Note that the actual_checking_round = patience * cfg.eval.freq
    # To disable the early stop, set the early_stop.patience a integer <=0
    cfg.early_stop.patience = 5
    # delta (float): Minimum change in the monitored metric to indicate an improvement.
    cfg.early_stop.delta = 0.0
    # Early stop when no improve to last `patience` round, in ['mean', 'best']
    cfg.early_stop.improve_indicator_mode = 'best'
    cfg.early_stop.the_smaller_the_better = True

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_training_cfg)


def assert_training_cfg(cfg):
    if cfg.backend not in ['torch', 'tensorflow']:
        raise ValueError(
            "Value of 'cfg.backend' must be chosen from ['torch', 'tensorflow']."
        )
    if cfg.backend == 'tensorflow' and cfg.federate.mode == 'standalone':
        raise ValueError(
            "We only support run with distribued mode when backend is tensorflow"
        )
    if cfg.backend == 'tensorflow' and cfg.use_gpu is True:
        raise ValueError(
            "We only support run with cpu when backend is tensorflow")

    if cfg.trainer.finetune.before_eval is False and cfg.trainer.finetune.steps <= 0:
        raise ValueError(
            f"When adopting fine-tuning, please set a valid local fine-tune steps, got {cfg.trainer.finetune.steps}"
        )


register_config("fl_training", extend_training_cfg)
