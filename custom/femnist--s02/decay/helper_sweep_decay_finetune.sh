for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg custom/femnist--s02/base_finetune.yaml \
    outdir 'custom/femnist--s02/wandb' \
    wandb.use True \
    wandb.name_user 'joelavond' \
    seed 2
    expname femnist--s02--n_epochs${local_update_steps}--lr${lr}--beta${beta}--exact--seed2 \
    federate.method 'decay' \
    trainer.model_on_batch_or_epoch 'batch' \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr} \
    trainer.beta ${beta}

