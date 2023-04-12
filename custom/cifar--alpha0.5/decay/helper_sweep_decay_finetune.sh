for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg custom/cifar--alpha0.5/base_finetune.yaml \
    outdir 'custom/cifar--alpha0.5/wandb' \
    wandb.use True \
    wandb.name_project 'exact_decay' \
    wandb.name_user 'joelavond' \
    expname cifar--alpha0.5--n_epochs${local_update_steps}--lr${lr}--beta${beta}--exact \
    federate.method 'exact_decay' \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr} \
    trainer.beta ${beta}

