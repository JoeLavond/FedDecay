## FedDecay
This repo holds the code to reproduce all experimental results found in the paper: \
FedDecay: Adapting to Data Heterogeneity in Federated Learning With Gradient Decay. 

Forked from [FederatedScope](https://github.com/alibaba/FederatedScope/tree/Feature/pfl_bench) (https://github.com/alibaba/FederatedScope/tree/Feature/pfl_bench). \
See original repo for more detailed instillation instructions and documentation.

<!--
Remark: FederatedScope has implemented learning rate scheduling since the version forked for our experiments.
We suggested that future use of FedDecay should be re-written directly as learning rate scheduling.
For iteration $t$, the learning rate should be set as $\eta_t = \eta \times \beta_{t % K}$
where $\eta$ is the initial learning rate, 
$K$ is the number of local update steps, 
and $\beta_{t % K}$ is the local decay factor. 
-->

### Implementation

Custom trainers were added at:
- `federatedscope/core/trainers/trainer_decay.py`
- `federatedscope/core/trainers/trainer_FOMAML.py`

Since learning rate scheduling was not yet implemented in forked version of FederatedScope, 
the implementation of FedDecay and FOMAML are based off storing the model weights at the start and end of each local update. 
The model weights are then overwritten with an interpolation of the stored weights and the current weights 
consistent with the model returned if the learning rate was properly decayed.

Added FedDecay and FOMAML configs at `federatedscope/core/configs/cfg_training.py` \
New configs:
- `cfg.training.decay_scheme` (default: 'exponential') - Whether to use exponential or linear decay
- `cfg.training.beta` - Decay factor
- `cfg.training.finetune_beta` (default: 1.0) - Do not decay the learning rate during finetuning
- `cfg.training.model_on_batch_or_epoch` (default: 'epoch') - How often to decay the learning rate

Small change to `federatedscope/core/auxileries/trainer_builder.py` to include the new trainers.

### Hyperparameter Sweeps
All hyperparameter sweeps were run using [Weights and Biases](https://wandb.ai/site) (https://wandb.ai/site). \
The sweep configurations can be found in the `feddecay` folder. \
Inside there is a subfolder for each data set which contains:
- `base_finetune.yaml` - the base configuration for the data set
- `<method_name>/sweep_<method_name>_finetune.yaml` - specifies the grid search for the method
- `<method_name>/helper_sweep_<method_name>_finetune.yaml` - runs the sweep for the method

Run the following command to initialize a sweep and return its sweep ID.
```bash
wandb sweep feddecay/<data_set>/<method_name>/helper_sweep_<method_name>_finetune.yaml
```

Run the following command (many times) to launch a process to execute the sweeps for each time run. \
Specific GPUs can be specified by changing the `CUDA_VISIBLE_DEVICES` environment variable.
```bash
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>
```

Supplementary experiments can be produced with the additional arguments:
- seed - to set the random seed
- training.decay_scheme 'linear' - to use linear decay

For example, add a line with `training.decay_scheme 'linear' /\` to the configurations specified in `feddecay/<data_set>/decay/helper_sweep_decay_finetune.sh` 