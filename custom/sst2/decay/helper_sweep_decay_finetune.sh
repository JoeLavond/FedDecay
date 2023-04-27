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
    wandb.name_user 'joelavond' \
    seed 2
    expname sst2--n_epochs${local_update_steps}--batch_size${batch_size}--lr${lr}--beta${beta}--exact--seed2 \
    federate.method 'decay' \
    # trainer.model_on_batch_or_epoch 'batch' \
    federate.local_update_steps ${local_update_steps} \
    data.batch_size ${batch_size} \
    optimizer.lr ${lr} \
    trainer.beta ${beta}

