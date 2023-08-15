for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg custom/femnist--s02/base.yaml \
    outdir 'custom/femnist--s02/wandb' \
    wandb.use True \
    wandb.name_user 'joelavond' \
    seed 0
    expname femnist--s02--fedavg--n_epochs${local_update_steps}--lr${lr}--seed0 \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr}

