<div align="center">

# Profile Based Anomaly Detection in Service Oriented Business Processes

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

Repository to support the master thesis "Profile-based Anomaly Detection in Service Oriented Business Processes"

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/nico-ru/profile-based-anomaly-detection
cd profile-based-anomaly-detection

# [OPTIONAL] create conda environment
conda create -n <myenv> python=3.10
conda activate <myenv>

# install pytorch according to instructions
# https://pytorch.org/get-started/
# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu
# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [config/experiment/](config/experiment/)

```bash
python src/train.py experiment=<experiment>.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 
```
