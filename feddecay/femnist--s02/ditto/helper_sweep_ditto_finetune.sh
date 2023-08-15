for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg feddecay/femnist--s02/base_finetune.yaml \
    outdir 'feddecay/femnist--s02/wandb' \
    wandb.use True \
    federate.method 'Ditto' \
    expname femnist--s02--n_epochs${local_update_steps}--lr${lr}--regular_weight${regular_weight} \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr} \
    personalization.regular_weight ${regular_weight}


