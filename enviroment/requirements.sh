#
. ../bootstrap.sh

#
yes | conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia;
yes | conda install pyg -c pyg;
yes | conda install -c huggingface -c conda-forge transformers tokenizers datasets

#
yes | conda install pip;
yes | pip install wandb;
yes | pip install pympler;
yes | pip install yacs;
yes | pip install rdkit;
yes | pip install python-louvain;
