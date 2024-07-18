#!/usr/bin/env bash

device="1"

#config="config_yaml/CLIP_Adapter/CLIP_adapter_cifar100.yaml"
config="config_yaml/CLIP_local_fe/imagenet_r.yaml"



cd ~/projects/My_CL_Bank/code || exit
conda activate pytorchEnv
CUDA_VISIBLE_DEVICES=$device python main.py --yaml_path=$config
