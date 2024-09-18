#!/usr/bin/env bash

device="3"

#config="config_yaml/CLIP_Adapter/CLIP_adapter_imagenet_r.yaml"
#config="config_yaml/CLIP_Adapter/CLIP_adapter_cifar100.yaml"
#config="config_yaml/CLIP_Adapter/CLIP_adapter_skin40.yaml"
#config="config_yaml/CLIP_Adapter/CLIP_adapter_cub200.yaml"
config="config_yaml/CLIP_Adapter/CLIP_adapter_cars196.yaml"



cd ~/projects/My_CL_Bank/code || exit
conda activate pytorchEnv
CUDA_VISIBLE_DEVICES=$device python main.py --yaml_path=$config
