import os
import sys
import torch
from methods import get_method
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(config, run_dir, log_mode="w"):
    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s => %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if config.is_log:
        log_file = os.path.join(run_dir, config.version_name + ".log")
        if os.path.exists(log_file):
            x = input("log file exists, input yes to rewrite:")
            if x == "yes" or x == "y":
                log_permission = True
            else:
                log_permission = False
        else:
            log_permission = True
        if log_permission:
            file_handler = logging.FileHandler(filename=log_file, mode=log_mode)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.info("logger created!")
    return logger


if __name__ == '__main__':
    is_train = True
    config = config()
    config.pretrained_path = os.path.join(os.environ["HOME"], config.pretrained_path) if config.pretrained_path else ""
    run_dir = ckpt_path = config.save_path + "/" + config.method + "/" + config.version_name
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    config.run_dir = run_dir
    if not is_train:
        config.is_log = False
        config.save_checkpoint = False
        logger = log(config, run_dir)
        os.environ["WANDB_DISABLED"] = "true"
        torch.set_num_threads(config.num_workers)  # limit cpu usage, important for DarkER, X-DER
        set_random(config.random_seed)
        data_manager = DataManager(config, logger)
        method_class = get_method(config.method)
        trainer = method_class(config, logger)

        for task_id in range(data_manager.num_tasks):
            logger.info("="*100)
            trainer.update_class_num(task_id)
            trainer.prepare_task_data(data_manager, task_id, is_train)
            task_checkpoint = torch.load(os.path.join(ckpt_path, f"checkpoint_task{task_id}.pkl"))
            trainer.prepare_model(task_id, task_checkpoint)
            trainer.eval_task(task_id)
            # trainer.after_task(task_id)
            logger.info("=" * 100)

        del trainer

    elif is_train:
        resume = False
        logger = log(config, run_dir, log_mode="a" if resume else "w")
        logger.info("config: {}".format(vars(config)))
        os.environ["WANDB_DISABLED"] = "false"
        os.environ["WANDB_MODE"] = "offline"

        # torch.set_num_threads(config.num_workers)  # limit cpu usage, important for DarkER, X-DER

        set_random(config.random_seed)
        wandb.init(
            # set the wandb project where this run will be logged
            project="CL_Bank",
            name=config.version_name,
            # id=version_name,
            dir=os.environ["HOME"],
            resume=False,
            # track hyperparameters and run metadata
            config=vars(config)
        )

        data_manager = DataManager(config, logger)

        method_class = get_method(config.method)
        trainer = method_class(config, logger)

        for task_id in range(len(config.increment_steps)):
            logger.info("="*100)
            trainer.update_class_num(task_id)
            trainer.prepare_task_data(data_manager, task_id)
            if os.path.exists(os.path.join(ckpt_path, f"checkpoint_task{task_id}.pkl")) and resume:
                task_ckpt = torch.load(os.path.join(ckpt_path, f"checkpoint_task{task_id}.pkl"))
                trainer.prepare_model(task_id, checkpoint=task_ckpt)
            else:
                trainer.prepare_model(task_id)
                trainer.incremental_train(data_manager, task_id)
            trainer.update_memory(data_manager)
            trainer.eval_task(task_id)
            trainer.after_task(task_id)
            logger.info("=" * 100)

        del trainer
    torch.cuda.empty_cache()

