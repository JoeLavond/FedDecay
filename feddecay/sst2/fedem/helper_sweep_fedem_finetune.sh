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
    wandb.name_project 'decay' \
    federate.method 'FedEM' \
    model.model_num_per_trainer 3 \
    expname sst2--n_epochs${local_update_steps}--batch_size${batch_size}--lr${lr} \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr} \
    data.batch_size ${batch_size}


