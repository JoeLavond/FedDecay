for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg custom/sst2/base_finetune.yaml \
    outdir 'custom/sst2/wandb' \
    wandb.use True \
    wandb.name_project 'decay' \
    wandb.name_user 'joelavond' \
    expname sst2--n_epochs${local_update_steps}--lr${lr}--beta${beta}--finetune_lr${ft_lr}--exact \
    federate.method 'decay' \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr} \
    trainer.beta ${beta} \
    trainer.finetune.lr ${ft_lr}

