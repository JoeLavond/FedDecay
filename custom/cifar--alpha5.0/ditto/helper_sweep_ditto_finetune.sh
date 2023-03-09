for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg custom/cifar--alpha5.0/base_finetune.yaml \
    outdir 'custom/cifar--alpha5.0/wandb' \
    wandb.use True \
    wandb.name_project 'decay' \
    wandb.name_user 'joelavond' \
    federate.method 'Ditto' \
    expname cifar--alpha5.0--n_epochs${local_update_steps}--lr${lr}--regular_weight${regular_weight} \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr} \
    personalization.regular_weight ${regular_weight}


