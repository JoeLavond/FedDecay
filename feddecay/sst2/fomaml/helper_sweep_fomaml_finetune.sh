for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg feddecay/sst2/base_finetune.yaml \
    outdir 'feddecay/sst2/wandb' \
    wandb.use True \
    expname sst2--fomaml--n_epochs${local_update_steps}--batch_size${batch_size}--lr${lr} \
    federate.method 'fomaml' \
    federate.local_update_steps ${local_update_steps} \
    data.batch_size ${batch_size} \
    optimizer.lr ${lr}

