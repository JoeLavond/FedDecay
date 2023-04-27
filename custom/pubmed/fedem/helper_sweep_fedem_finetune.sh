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
    wandb.name_user 'joelavond' \
    federate.method 'FedEM' \
    federate.sample_client_num 3 \
    federate.unseen_clients_rate 0.4 \
    model.model_num_per_trainer 3 \
    expname pubmed--fedem--n_epochs${local_update_steps}--lr${lr} \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr}


