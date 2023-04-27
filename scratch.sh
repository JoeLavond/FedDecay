#!/bin/bash
# print execute info
echo "";
current_date_time=$(date)
echo "last execute at $current_date_time";
echo "";

# control
screen_count_max=3
gpu_number=2

# get number of detached screens
screen_count=$(screen -ls | grep Detached | awk 'END{print NR}')

# count python processes on GPU "gpu_id"
gpu2_count=$(nvidia-smi -i 2 | grep 'python' | awk 'END{print NR}')
gpu3_count=$(nvidia-smi -i 3 | grep 'python' | awk 'END{print NR}')

condition=$(($gpu_number * $screen_count_max))
if [[ $gpu2_count -lt $screen_count_max && $screen_count -lt $condition ]]
then
    screen -dm bash -c "
        source activate fs;
        CUDA_VISIBLE_DEVICES=2 wandb agent joelavond/decay--extras/pdlgw13t;
    ";
fi

if [[ $gpu3_count -lt $screen_count_max && $screen_count -lt $condition ]]
then
    screen -dm bash -c "
        source activate fs;
        CUDA_VISIBLE_DEVICES=3 wandb agent joelavond/decay--extras/m4eccgum;
    ";
fi

# descriptives
nvidia-smi
#screen -ls
echo "$gpu2_count and $gpu3_count screens are currently running on GPUs 2 and 3 of $screen_count screens, respectively.";
echo "This meets or exceeds the individual screen count max of $screen_count_max";

