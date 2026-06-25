# FedDecay

Balancing Model Performance and Rapid Personalization in Federated Learning with Learning Rate Scheduling  
Authors: Joseph Lavond, Minhao Cheng, and Yao Li

## Overview

This repository contains the code for our paper "FedDecay: Balancing Model Performance and Rapid Personalization in Federated Learning with Learning Rate Scheduling".

We propose FedDecay, a method for improving federated learning on heterogeneous data by applying a within-round learning-rate decay to each client's local update steps.
Later local steps, which drift furthest from the global model, are attenuated by a decay factor β; earlier steps, which stay close to the global optimum, are kept at full strength.
The result is a better-aligned global update with no change to the communication protocol.

This repository is a fork of [FederatedScope (pfl\_bench branch)](https://github.com/alibaba/FederatedScope/tree/Feature/pfl_bench).
Three files were added and two files were modified relative to the upstream codebase.

> **Implementation note:** FederatedScope added native learning-rate scheduling support after the version used for these experiments was forked. Future implementations of FedDecay should use that scheduler instead. At local step $t$ within a round of $K$ steps, set the learning rate as $\eta_t = \eta \times \beta^{(t \bmod K)}$ where $\eta$ is the base learning rate and $\beta$ is the decay factor.

## Getting Started

Here we provide a step-by-step guide to set up the environment for our experiments.
Our code requires **Python 3.9** and uses conda for environment management.
See the [pfl\_bench README](https://github.com/alibaba/FederatedScope/tree/Feature/pfl_bench) for additional FederatedScope documentation.

1. Clone the repository

```bash
git clone https://github.com/JoeLavond/FedDecay.git --branch bench
cd FedDecay
```

1. Create and activate a conda environment

```bash
conda create -n feddecay python=3.9
conda activate feddecay
```

1. Install dependencies

```bash
# PyTorch with CUDA (adjust cuda version to match your driver)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# PyTorch Geometric (required for PubMed graph dataset)
conda install pyg -c pyg

# HuggingFace Transformers (required for SST2 NLP dataset)
conda install -c huggingface -c conda-forge transformers tokenizers datasets

# Remaining dependencies
conda install pip
pip install wandb pympler yacs rdkit python-louvain
```

1. Install the package

```bash
pip install -e .
```

1. Log in to Weights & Biases (used for hyperparameter sweeps)

```bash
wandb login
```

## Usage

All experiments are run using Weights & Biases sweeps.
Each dataset/method combination has its own sweep configuration under `feddecay/<dataset>/<method>/`.

To run FedDecay on FEMNIST:

```bash
# Initialise a sweep (returns a sweep ID)
wandb sweep feddecay/femnist--s02/decay/sweep_decay_finetune.yaml

# Launch a sweep agent
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>
```

Run the agent command in multiple terminals or on multiple machines to parallelise the sweep.
Each agent picks a configuration from the W&B server, runs it using `helper_sweep_<method>_finetune.sh`, and reports results back.

Common FedDecay command line arguments include:

- `--federate.method`: set to `decay` to enable FedDecay (or `FedAvg`, `Ditto`, etc. for baselines)
- `--trainer.beta`: decay factor β ∈ (0, 1]; 1.0 is equivalent to FedAvg
- `--trainer.decay_scheme`: `exponential` (default) or `linear`
- `--federate.local_update_steps`: number of local epochs K per round
- `--optimizer.lr`: base learning rate η

The paper's grid search covered `beta ∈ {0.2, 0.4, 0.6, 0.8}` with `local_update_steps = 3` and `lr ∈ {0.005, 0.01, 0.05, 0.1, 0.5}` on each dataset.

## Code Structure

There are two main directories added by FedDecay: `./feddecay` and additions to `./federatedscope`.

The `./feddecay` directory contains W&B sweep configurations for every dataset × method combination used in the paper's experiments.
Each dataset folder contains one sub-folder per method: `decay`, `fedavg`, `ditto`, `pfedme`, `fedem`, `fedbn`, `fomaml`.
The main FedDecay implementation lives in `./federatedscope/core/trainers/trainer_decay.py`.

The full code structure for FedDecay-specific files is as follows:

```
├── feddecay/                       # sweep configurations
│   ├── femnist--s02/               # FEMNIST (natural author-level heterogeneity)
│   ├── sst2/                       # SST2 sentiment, Dirichlet α = 0.4
│   ├── pubmed/                     # PubMed node classification
│   ├── cifar10--alpha5.0/          # CIFAR-10 supplementary experiment
│   │   ├── <method>/               # one folder per method
│   │   │   ├── base_finetune.yaml          # full FederatedScope config
│   │   │   ├── sweep_<method>_finetune.yaml    # W&B sweep definition
│   │   │   ├── helper_sweep_<method>_finetune.sh   # shell script for sweep agents
│
├── federatedscope/core/trainers/
│   ├── trainer_decay.py            # FedDecay trainer (added)
│   ├── trainer_FOMAML.py           # First-Order MAML baseline (added)
│
├── federatedscope/core/
│   ├── configs/cfg_training.py     # added beta, decay_scheme, and related keys (modified)
│   ├── auxiliaries/trainer_builder.py  # dispatch for decay and FOMAML methods (modified)
```

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{
lavond2025feddecay,
title={FedDecay: Balancing Model Performance and Rapid Personalization in Federated Learning with Learning Rate Scheduling},
author={Joseph Lavond and Minhao Cheng and Yao Li},
year={2025},
url={https://github.com/JoeLavond/FedDecay}
}
```
