#!/bin/bash
# Execute a list of commands from $file_name \
#   on the GPUs specified in $gpu_ids \
#   $max_gpu_experiment_count at a time \
# Warning: this script will launch experiments based off only what the current user is running
## Instructions
# This script executes the commands in the following file
# It will run the following number of commands on the specified GPUs when available
# An example line in the file is: python gan_train.py

## Default values
unset -v sweep_id
conda_environment='fs'
gpu_offset=0
num_gpu=4
max_gpu_experiment_count=1
use_tmux=false
source_or_conda='source'
debug=false

use_str=$(cat <<-END
    Usage: $0
        [-w <sweep_id>]
        [-e <conda_environment> (default: $conda_environment)]
        [-o <gpu_offset> (default: $gpu_offset)]
        [-n <num_gpu> (default: $num_gpu)]
        [-m <max_gpu_experiment_count> (default: $max_gpu_experiment_count)]"
        [-t <use_tmux> (default: $use_tmux)]
        [-s <source_or_conda> (default: $source_or_conda)]
        [-d <debug> (default: $debug)]
END

)

usage() {
    echo "$use_str"
}


## Command line arguments
while [ $# -gt 0 ]
do
  case "$1" in
    -w) shift; sweep_id="$1";;
    -e) shift; conda_environment="$1";;
    -o) shift; gpu_offset="$1";;
    -n) shift; num_gpu="$1";;
    -m) shift; max_gpu_experiment_count="$1";;
    -t) shift; use_tmux="$1";;
    -s) shift; source_or_conda="$1";;
    -d) shift; debug="$1";;
    -h) usage; exit 1;;
    *) usage; exit 1;;
  esac
  shift
done

if [ -z "$sweep_id" ]
then
    echo "Error: need to specific sweep_id with -w flag."
    exit 1
fi
echo "Running: $0 -w $sweep_id -e $conda_environment -o $gpu_offset -n $num_gpu -m $max_gpu_experiment_count -t $use_tmux -s $source_or_conda -d $debug"

# establish gpu indices 0, 1, .., num_gpu
gpu_ids=()
for ((i=$gpu_offset; i<$gpu_offset + $num_gpu; i++))
do
    gpu_ids+=("$i")
done

## Assertions
# check tmux installation
if "$use_tmux"
then
    if ! command -v tmux &> /dev/null
    then
      echo "Warning: tmux could not be found. Attempting to run using screen."
      use_tmux=false
    fi
fi

if ! "$use_tmux"
then
    if ! command -v screen &> /dev/null
    then
        echo "Error: tmux and screen could both not be found. Please check for installation."
        exit 1
    fi
fi

# require conda to be installed and the specified environment to exist
if command -v conda &> /dev/null
then
  if ! conda env list | grep -q "$conda_environment"
  then
    echo "Error: conda environment $conda_environment could not be found."
    exit 1
  fi
else
  echo "Error: conda could not be found. Please check for installation."
  exit 1
fi

## Main loop
command="wandb agent joelavond/$sweep_id"
if "$debug"
then
    command="wandb agent joelavond/$sweep_id; sleep 300;"
fi

command_bash="$source_or_conda activate $conda_environment; CUDA_VISIBLE_DEVICES=${gpu_ids[$gpu_index]} $command"

for exp_num in $(seq 1 1 "$max_gpu_experiment_count")
do

    for gpu_id in "${gpu_ids[@]}"
    do

        echo "Running $command on GPU $gpu_id"
        command_bash="$source_or_conda activate $conda_environment; CUDA_VISIBLE_DEVICES=$gpu_id $command"

        if "$use_tmux"
        then
            if "$debug"
            then
                echo "tmux new-session -d bash -c $command_bash"
            fi
            tmux new-session -d "bash -c '$command_bash'"
        else
            if "$debug"
            then
                echo "screen -dm bash -c $command_bash"
            fi
            screen -dm bash -c "$command_bash"
        fi

    done
done

