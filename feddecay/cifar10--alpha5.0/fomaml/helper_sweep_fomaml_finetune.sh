for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg feddecay/cifar10--alpha5.0/base_finetune.yaml \
    outdir 'feddecay/cifar10--alpha5.0/wandb' \
    wandb.use True \
    expname cifar10--alpha5.0--fomaml--n_epochs${local_update_steps}--lr${lr} \
    federate.method 'fomaml' \
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr}

