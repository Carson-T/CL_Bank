#!/usr/bin/env bash

device="2"
#config="config_yaml/PRCLIP/prclip_imagenet_r.yaml"
config="config_yaml/PRCLIP/prclip_cifar100.yaml"
#config="config_yaml/PRCLIP/prclip_imagenet100.yaml"
#config="config_yaml/PRCLIP/prclip_skin40.yaml"
#config="config_yaml/Proof/proof_cifar100.yaml"
#config="config_yaml/SLCA/SLCA_imagenet_r.yaml"

cd ~/projects/My_CL_Bank/code || exit
conda activate pytorchEnv
CUDA_VISIBLE_DEVICES=$device python main.py --yaml_path=$config
