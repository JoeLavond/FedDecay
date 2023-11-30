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
    expname femnist--s02--n_epochs${local_update_steps}--lr${lr}--beta${beta} \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr} \
    federate.method 'decay' \
    trainer.beta ${beta}

