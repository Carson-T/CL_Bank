#!/usr/bin/env bash

device="6"
#config="config_yaml/PRCLIP/prclip_imagenet_r.yaml"
#config="config_yaml/Proof/proof_cifar100.yaml"
config="config_yaml/SLCA/SLCA_cifar100.yaml"

CUDA_VISIBLE_DEVICES=$device python main.py --yaml_path=$config
