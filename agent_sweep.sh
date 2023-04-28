# screen -dm bash -c "
    # source activate fs;
    # CUDA_VISIBLE_DEVICES=0 wandb agent joelavond/${1};
# "
#
# screen -dm bash -c "
    # source activate fs;
    # CUDA_VISIBLE_DEVICES=1 wandb agent joelavond/${1};
# "

screen -dm bash -c "
    source activate fs;
    CUDA_VISIBLE_DEVICES=2 wandb agent joelavond/${1};
"

screen -dm bash -c "
    source activate fs;
    CUDA_VISIBLE_DEVICES=3 wandb agent joelavond/${1};
"
