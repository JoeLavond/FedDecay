## FedDecay

Code repository for the paper:

> **FedDecay: Adapting to Data Heterogeneity in Federated Learning With Gradient Decay**

FedDecay improves federated learning on heterogeneous data by applying a
within-round learning-rate decay to each client's local update steps.  Later
local steps, which drift furthest from the global model, are attenuated by a
decay factor β; earlier steps, which stay close to the global optimum, are
kept at full strength.  The result is a better-aligned global update with no
change to the communication protocol.

This repository is a fork of
[FederatedScope (pfl\_bench branch)](https://github.com/alibaba/FederatedScope/tree/Feature/pfl_bench).
Three files were added and two files were modified relative to the upstream
codebase (see [Changes from upstream](#changes-from-upstream)).

> **Implementation note:** FederatedScope added native learning-rate
> scheduling support after the version used for these experiments was forked.
> Future implementations of FedDecay should use that scheduler instead.
> At local step $t$ within a round of $K$ steps, set the learning rate as
> $\eta_t = \eta \times \beta^{(t \bmod K)}$ where $\eta$ is the base
> learning rate and $\beta$ is the decay factor.

---

## Getting Started

These instructions walk through setting up a clean conda environment, installing
the package, and launching a hyperparameter sweep to reproduce the paper results.
See the [pfl\_bench README](https://github.com/alibaba/FederatedScope/tree/Feature/pfl_bench)
for additional FederatedScope documentation.

### Prerequisites

- [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- A [Weights & Biases](https://wandb.ai/site) account (free) for hyperparameter sweeps
- A GPU is recommended for the NLP experiments (SST2 / BERT); CPU is sufficient for FEMNIST

### Step 1 — Clone the repository

```bash
git clone https://github.com/JoeLavond/FedDecay.git --branch bench
cd FedDecay
```

### Step 2 — Create the conda environment

Create a fresh conda environment and install the base FederatedScope
dependencies.  The `requirements-feddecay.sh` script installs the packages
needed on top of a standard PyTorch conda install:

```bash
# Create and activate a Python 3.9 environment
conda create -n feddecay python=3.9
conda activate feddecay

# Install PyTorch with CUDA (adjust cuda version to match your driver)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 \
    -c pytorch -c nvidia

# Install PyTorch Geometric (required for the PubMed graph dataset)
conda install pyg -c pyg

# Install HuggingFace Transformers (required for the SST2 NLP dataset)
conda install -c huggingface -c conda-forge transformers tokenizers datasets

# Install remaining dependencies
conda install pip
pip install wandb pympler yacs rdkit python-louvain
```

Alternatively, install from the environment file provided in the upstream
pfl\_bench repo:

```bash
conda install --file enviroment/requirements-torch1.10-application.txt \
    -c pytorch -c conda-forge -c nvidia -c pyg
```

### Step 3 — Install the package

From the root of the cloned repository:

```bash
pip install -e .
```

The `-e` (editable) flag means Python will use the source files directly,
so any changes you make to the FedDecay trainer code take effect immediately
without reinstalling.

### Step 4 — Log in to Weights & Biases

```bash
wandb login
```

Follow the prompts to authenticate.  Sweep results are stored in your W&B
workspace under the project names defined in each sweep YAML (e.g.
`decay--femnist--s02`).

### Step 5 — Prepare datasets

FederatedScope's DataZoo downloads and preprocesses datasets automatically
the first time a run references them.  No manual download is required for
FEMNIST, PubMed, or SST2; they are fetched from the internet on first use
and cached under the `data/` directory.

For FEMNIST and PubMed the data is small and downloads in seconds.  SST2
uses a small BERT model (`google/bert_uncased_L-2_H-128_A-2`) which is
downloaded from HuggingFace on first use and cached under `transformers/`.

### Step 6 — Run a hyperparameter sweep

All experiments were run using Weights & Biases sweeps.  Each dataset/method
combination has its own sweep configuration under `feddecay/<dataset>/<method>/`.

**Initialise a sweep** (returns a sweep ID):

```bash
wandb sweep feddecay/<dataset>/<method>/sweep_<method>_finetune.yaml
```

For example, to sweep FedDecay on FEMNIST:

```bash
wandb sweep feddecay/femnist--s02/decay/sweep_decay_finetune.yaml
```

**Launch one or more sweep agents** (each agent runs a single configuration):

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>
```

Run this command in multiple terminals (or on multiple machines) to
parallelise the sweep.  Each agent picks a configuration from the W&B
server, runs it using `helper_sweep_<method>_finetune.sh`, and reports
results back.

**Supplementary experiments** (random seeds, linear decay) can be run by
appending extra arguments.  For example, to run FedDecay with linear decay:

```bash
# Edit helper_sweep_decay_finetune.sh and add:
#     training.decay_scheme 'linear' \
# to the argument list, then re-run as above.
```

---

## Datasets and Sweep Configurations

| Folder | Dataset | Notes |
|---|---|---|
| `feddecay/femnist--s02/` | FEMNIST | Natural author-level heterogeneity |
| `feddecay/sst2/` | SST2 (sentiment) | Dirichlet α = 0.4 heterogeneity |
| `feddecay/pubmed/` | PubMed (graph) | Node classification on citation graph |
| `feddecay/cifar10--alpha5.0/` | CIFAR-10 | Supplementary experiment |

Each dataset folder contains one sub-folder per method:
`decay`, `fedavg`, `ditto`, `pfedme`, `fedem`, `fedbn`, `fomaml`.

Each method sub-folder contains:
- `base_finetune.yaml` — full FederatedScope configuration for this dataset
- `sweep_<method>_finetune.yaml` — W&B sweep definition (grid search parameters)
- `helper_sweep_<method>_finetune.sh` — shell script called by each W&B agent

---

## Changes from Upstream

Three files were **added** and two files were **modified** relative to the
[FederatedScope pfl\_bench branch](https://github.com/alibaba/FederatedScope/tree/Feature/pfl_bench).

### Files added

**`federatedscope/core/trainers/trainer_decay.py`**

Implements the FedDecay trainer as a hook-based wrapper over any existing
FederatedScope trainer type.  Before each local update step the current model
weights are saved; after the step the weight change is scaled by the decay
factor appropriate for that step index and written back.  This is equivalent
to multiplying the learning rate by β^k at local step k (exponential) or
max(0, 1 − β·k) (linear).  The file contains full docstrings and inline
comments explaining the mathematics.

**`federatedscope/core/trainers/trainer_FOMAML.py`**

Implements a First-Order MAML baseline.  FOMAML uses only the last local
gradient step (θ_K − θ_{K−1}) as the meta-gradient approximation, avoiding
the need for second-order (Hessian) information.  Snapshots are taken at
local steps K−1 and K; the final update anchors the result at θ_0 so the
server aggregation receives a direction equivalent to a single meta-gradient
step.

**`feddecay/`**

W&B sweep configurations for every dataset × method combination used in
the paper's experiments.

### Files modified

**`federatedscope/core/configs/cfg_training.py`**

Four new configuration keys were added under `cfg.trainer`:
- `cfg.trainer.model_on_batch_or_epoch` (default `'epoch'`) — decay granularity
- `cfg.trainer.beta` (default `1.0`) — within-round decay factor β
- `cfg.trainer.finetune_beta` (default `1.0`) — β used during fine-tuning
- `cfg.trainer.decay_scheme` (default `'exponential'`) — schedule type

With all defaults, the behaviour is identical to vanilla FedAvg (β = 1 → no decay).

**`federatedscope/core/auxiliaries/trainer_builder.py`**

Two `elif` branches were added to `get_trainer()` to dispatch
`federate.method = 'decay'` to `wrap_decay` and
`federate.method = 'FOMAML'` to `wrap_FOMAML`.

---

## Configuration Reference

The key FedDecay parameters in a sweep YAML or base config are:

| Key | Type | Default | Description |
|---|---|---|---|
| `federate.method` | string | `'FedAvg'` | Set to `'decay'` to enable FedDecay |
| `trainer.beta` | float | `1.0` | Decay factor β ∈ (0, 1]; 1.0 = FedAvg |
| `trainer.decay_scheme` | string | `'exponential'` | `'exponential'` or `'linear'` |
| `trainer.model_on_batch_or_epoch` | string | `'epoch'` | Decay granularity |
| `trainer.finetune_beta` | float | `1.0` | β during fine-tuning (usually 1.0) |
| `federate.local_update_steps` | int | — | Number of local epochs K per round |
| `optimizer.lr` | float | `0.1` | Base learning rate η |

The paper's grid search covered `beta ∈ {0.2, 0.4, 0.6, 0.8}` with
`local_update_steps = 3` and `lr ∈ {0.005, 0.01, 0.05, 0.1, 0.5}` on each
dataset, using W&B Hyperband early termination (`min_iter: 3`).

---

## Implementation Notes

Since learning-rate scheduling was not yet implemented in the version of
FederatedScope that was forked, FedDecay achieves the equivalent effect at
the *weight level* rather than the learning-rate level:

1. Before each local update step: save a deep copy of the model as
   `ctx.last_model`.
2. Run the standard optimiser step (updating `ctx.model` in place).
3. Compute the weight change Δθ = ctx.model − ctx.last\_model.
4. Scale: Δθ\_scaled = scale · Δθ, where scale = β^k (exponential) or
   max(0, 1 − β·k) (linear), and k is the 0-based step index within the round.
5. Restore: ctx.model ← ctx.last\_model + Δθ\_scaled.

Steps 3–5 are mathematically equivalent to running the optimiser with a
learning rate of η · scale at step k.

With β = 1 (the default), scale = 1 at every step and the behaviour is
identical to vanilla FedAvg.
