import torch
import yaml
import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', default="config_yaml/DA/DA_cifar100.yaml")

    parser.add_argument('--device_ids', default=None)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--increment_type', default=None)
    parser.add_argument('--increment_steps', type=list, default=None)
    parser.add_argument('--is_openset_test', type=bool, default=False)

    parser.add_argument('--dataset_name', default=None)
    parser.add_argument('--data_shuffle', default=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=None)
    # parser.add_argument('--Blur', type=int, default=1)
    # parser.add_argument('--OGE', type=int, default=0)
    # parser.add_argument('--CLAHE', type=int, default=1)
    # parser.add_argument('--Cutout', type=int, default=1)
    parser.add_argument('--method', default=None)
    parser.add_argument('--backbone', default=None)  # resnet18.a1_in1k
    parser.add_argument('--pretrained_path', default=None)  # ../../../pretrained_model/resnet34.a1_in1k.safetensors
    parser.add_argument('--drop_rate', type=float, default=0)
    parser.add_argument('--drop_path_rate', type=float, default=0)

    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--optimizer', default=None)
    parser.add_argument('--loss_func', default="CEloss")
    parser.add_argument('--label_smoothing', type=float, default=0)
    parser.add_argument('--scheduler', default=None)  #Warm-up-Cosine-Annealing  multi_step
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--init_ratio', type=float, default=None)       # Warm-up-Cosine-Annealing parameters
    parser.add_argument('--min_lr_ratio', type=float, default=None)    # Warm-up-Cosine-Annealing parameters
    # parser.add_argument('--T', type=float, default=None)

    parser.add_argument('--milestones', type=list, default=None)
    parser.add_argument('--gamma', type=float, default=None)

    parser.add_argument('--memory_size', type=int, default=None)
    parser.add_argument('--memory_per_class', type=int, default=None)
    parser.add_argument('--sampling_method', default=None)

    parser.add_argument('--version_name', default=None)
    parser.add_argument('--save_path', default="../checkpoint&log")
    parser.add_argument('--is_log', default=True)
    parser.add_argument('--save_checkpoint', default=False)

    parser.add_argument('--apply_nme', type=bool, default=False)

    all_configs = parse_args_and_yaml(parser)

    return all_configs


def parse_args_and_yaml(parser):
    given_configs, _ = parser.parse_known_args()
    # print(given_configs)
    if given_configs.yaml_path:
        yaml_args = {}
        with open(given_configs.yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        yaml_args.update(cfg["basic"])
        yaml_args.update(cfg["usual"])
        scheduler = yaml_args["scheduler"]
        if "options" in cfg:
            yaml_args.update(cfg["options"][scheduler])
        if "special" in cfg:
            yaml_args.update(cfg["special"])
        parser.set_defaults(**yaml_args)
    all_configs = parser.parse_args()
    # print(all_configs)

    return all_configs


