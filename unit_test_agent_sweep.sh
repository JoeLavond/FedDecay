bash -c "
    source activate fs;
    CUDA_VISIBLE_DEVICES=0 wandb agent joelavond/decay/${1};
    sleep 300;
"

