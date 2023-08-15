#
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia;
conda install pyg -c pyg;
conda install -c huggingface -c conda-forge transformers tokenizers datasets

#
conda install pip;
pip install wandb;
pip install pympler;
pip install yacs;
pip install rdkit;
pip install python-louvain;
