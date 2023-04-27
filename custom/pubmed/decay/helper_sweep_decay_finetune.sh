for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python federatedscope/main.py \
    --cfg custom/pubmed/base_finetune.yaml \
    #
    # wandb
    outdir 'custom/pubmed/wandb' \
    wandb.use True \
    wandb.name_user 'joelavond' \
    #
    # expname
    expname pubmed--n_epochs${local_update_steps}--lr${lr}--beta${beta}--linear \
    #
    # basic tuning
    federate.local_update_steps ${local_update_steps} \
    optimizer.lr ${lr} \
    #
    # decay
    federate.method 'decay' \
    trainer.decay_scheme 'linear' \
    trainer.beta ${beta}

