#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings
import datetime

import numpy as np
import random

from time import sleep
from random import randint
from pytorch_metric_learning.utils import distributed as pml_dist
from pytorch_metric_learning import losses as pml_loss

from src.solver.adv_loss import CustomContrastiveLoss, SupContrastiveLoss
import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data.retrieval_loader import construct_retrieval_loader
from src.engine.evaluator import Evaluator_Submodel
from src.engine.trainer import Trainer, SubModelTrainer
from src.models.build_model import build_sub_models
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup

warnings.filterwarnings("ignore")

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup():
    """
    Create configs and perform basic setups.
    """
    args = default_argument_parser().parse_args()
    print(args)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # setup dist
    if args.local_rank != -1:
        assert cfg.NUM_GPUS > 1, "Requiring more than 1 GPUs!"
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=900))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.barrier()
    else:
        torch.cuda.set_device(0)
        
    # cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])

    # setup output dir
    # output_dir / data_name / expe_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    cfg_file_name = os.path.split(args.config_file)[-1].split(".")[0]
    output_folder = os.path.join(cfg.DATA.NAME, args.expe_name, "{:s}".format(cfg_file_name), f"lr{lr}_wd{wd}")
    eval_folder = os.path.join(cfg.EVAL.DATASET, args.expe_name, "{:s}".format(cfg_file_name), f"lr{lr}_wd{wd}")
    
    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        #eval_save_path = os.path.join(cfg.EVAL.SAVE_DIR, eval_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        #sleep(randint(1, 5))
        if not PathManager.exists(output_path):# and (not PathManager.exists(eval_save_path)):
            PathManager.mkdirs(output_path)
            #PathManager.mkdirs(eval_save_path)
            cfg.OUTPUT_DIR = output_path
            #cfg.EVAL.SAVE_DIR = eval_save_path
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")
    cfg.freeze()
    return cfg, args, args.local_rank

def get_loaders(cfg, logger):
    logger.info("Loading training data (final training data for landmark)...")
    train_loader, query_loader, gallery_loader, test_loader = construct_retrieval_loader(cfg)
    return train_loader, query_loader, gallery_loader, test_loader
    
def train(cfg, args, local_rank):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # fix the seed for reproducibility
    if cfg.SEED is not None:
        init_seeds(cfg.SEED + local_rank)

    # main training / eval actions here
    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("FCLearning")
    
    train_loader, query_loader, gallery_loader, test_loader = get_loaders(cfg, logger)
    
    logger.info("Constructing models...")

    if cfg.SOLVER.TYPE == 'submodel_compatible':
        sparsity = cfg.SUB_MODEL.SPARSITY
        model,cur_device = build_sub_models(cfg,sparsity)
    else:
        logger.warn(f'train type {cfg.SOLVER.TYPE}')
        raise NotImplementedError

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info("learnable parameter {:s}".format(name))
        else:
            logger.info("non-learnable parameter {:s}".format(name))
    
    criterion = {}
    if cfg.LOSS.TYPE in ['softmax', 'arcface', 'cosface']:
        criterion['base'] = torch.nn.CrossEntropyLoss()
    elif cfg.LOSS.TYPE == 'contra':
        criterion['base'] = CustomContrastiveLoss(cfg)
    elif cfg.LOSS.TYPE == "supcontra":
        criterion['base'] = SupContrastiveLoss(cfg)
    else:
        raise NotImplementedError

    logger.info("Setting up Evalutator...")

    evaluator = Evaluator_Submodel(cfg=cfg,device=cur_device,parent_model=model)

    logger.info("Setting up Trainer...")
    logger.info("number of training images: {:d}".format(int(len(train_loader.dataset))))
    
    # 梯度缩放器
    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP)
    if cfg.SOLVER.TYPE == "base":
        trainer = Trainer(cfg, model, criterion, grad_scaler, evaluator, cur_device, 
                          int(len(train_loader.dataset)), expe_name=args.expe_name)
    elif cfg.SOLVER.TYPE == 'submodel_compatible':
        trainer = SubModelTrainer(cfg, model, criterion, grad_scaler, 
                                    evaluator, cur_device, len(train_loader), 
                                    expe_name=args.expe_name)
    if train_loader:
        results = trainer.train(train_loader, query_loader, gallery_loader)
    else:
        print("No train loader presented. Exit")
    return results

def main():
    """main function to call from workflow"""
    # set up cfg and args
    cfg, args, local_rank = setup()

    # Perform training.
    results = train(cfg, args, local_rank)
    print(results)

if __name__ == '__main__':
    main()
