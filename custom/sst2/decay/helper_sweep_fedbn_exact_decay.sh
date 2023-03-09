for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg custom/sst2/base.yaml \
    outdir 'custom/sst2/wandb' \
    wandb.use True \
    wandb.name_user 'joelavond' \
    personalization.local_param "['bn', 'norms']"
    expname sst2--n_epochs${local_update_steps}--batch_size${batch_size}--lr${lr}--beta${beta}--fedbn--exact_decay \
    federate.method 'exact_decay' \
    federate.local_update_steps ${local_update_steps} \
    data.batch_size ${batch_size} \
    optimizer.lr ${lr} \
    trainer.beta ${beta}


