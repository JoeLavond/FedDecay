bash -c "
    source activate fs;
    CUDA_VISIBLE_DEVICES=0 wandb agent joelavond/${1};
    sleep 300;
"

