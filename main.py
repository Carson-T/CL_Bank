import os
import sys
from methods import get_method
import torch
from data_manager import DataManager
import numpy as np
import random
import logging
import wandb
from config import config


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log(config):
    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s => %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    run_dir = config.save_path+"/"+config.method+"/"+config.version_name
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    log_file = os.path.join(run_dir, config.version_name + ".log")
    if os.path.exists(log_file):
        x = input("log file exists, input yes to rewrite:")
        if x == "yes" or x == "y":
            log_permission = True
        else:
            log_permission = False
    else:
        log_permission = True
    if config.is_log and log_permission:
        file_handler = logging.FileHandler(filename=os.path.join(run_dir, config.version_name+".log"), mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("logger created!")
    return logger


if __name__ == '__main__':
    config = config()
    logger = log(config)
    logger.info("config: {}".format(vars(config)))
    # os.environ["WANDB_MODE"] = "offline"
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device_ids
    torch.set_num_threads(config.num_workers)  # limit cpu usage, important for DarkER, X-DER

    set_random(config.random_seed)
    wandb.init(
        # set the wandb project where this run will be logged
        project="CL_Bank",
        name=config.version_name,
        # id=version_name,
        dir=config.wandb_dir,
        resume=False,
        # track hyperparameters and run metadata
        config=vars(config)
    )

    data_manager = DataManager(config, logger)

    method_class = get_method(config.method)
    trainer = method_class(config, logger)
    # if config.method == "DER":
    #     trainer = Dynamic_ER(config, logger)
    # elif config.method == "iCarL":
    #     trainer = iCaRL(config, logger)
    # elif config.method == "L2P":
    #     trainer = L2P(config, logger)
    # elif config.method == "Dual_Prompt":
    #     trainer = Dual_Prompt(config, logger)
    # elif config.method == "Coda_Prompt":
    #     trainer = Coda_Prompt(config, logger)
    # elif config.method == "UCIR":
    #     trainer = UCIR(config, logger)
    # elif config.method == "Dark_ER":
    #     trainer = Dark_ER(config, logger)
    # elif config.method == "WA":
    #     trainer = WA(config, logger)
    # elif config.method == "Ease":
    #     trainer = Ease(config, logger)
    # else:
    #     raise ValueError("Unknown method!")

    for task_id in range(data_manager.num_tasks):
        logger.info("="*100)
        trainer.update_class_num(task_id)
        trainer.prepare_task_data(data_manager, task_id)
        trainer.prepare_model(task_id)
        trainer.incremental_train(data_manager, task_id)
        trainer.update_memory(data_manager)
        trainer.eval_task(task_id)
        trainer.after_task(task_id)
        logger.info("=" * 100)

    del trainer
    torch.cuda.empty_cache()
aaaaaa
