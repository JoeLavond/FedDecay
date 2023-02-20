for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg custom/pubmed/base_finetune.yaml \
    outdir 'custom/pubmed/wandb' \
    wandb.use True \
    wandb.name_project 'exact_decay' \
    wandb.name_user 'joelavond' \
    expname pubmed--n_epochs${local_update_steps}--n_batch${batch_size}--lr${lr}--beta${beta}--finetune_lr${ft_lr}--exact \
    federate.method 'exact_decay' \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr} \
    data.batch_size ${batch_size} ]
    trainer.beta ${beta} \
    trainer.finetune.lr ${ft_lr}

