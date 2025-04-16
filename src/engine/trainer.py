#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, collections
import numpy as np
import copy
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import torch.distributed as dist
from typing import List
from src.models.build_model import build_sub_models
from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer
from scipy.stats import entropy
from scipy.special import logsumexp
from ..solver.min_norm_solver import MinNormSolver
from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage, tensor_to_float
from ..utils.margin_softmax import large_margin_module
from ..solver.xbm import XBM
from ..solver.sfsc import TMC
from ..solver.adv_loss import gather_tensor
from torch.utils.tensorboard import SummaryWriter
import json
from timm.utils import NativeScaler
from tqdm import tqdm
import xlsxwriter as xw
import openpyxl
from ..models.subnetworks.subnet import SubnetConv2d, SubnetLinear
from ..models.subnetworks.slimmable_ops import SlimmableConv2d,SlimmableLinear
from ..models.subnetworks.resnet18 import GetSubnetFaster
from copy import deepcopy
logger = logging.get_logger("FCLearning")

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch, suffix=''):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries) + suffix)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return 'Iter:[' + fmt + '/' + fmt.format(num_batches) + ']'
    
class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self, 
        cfg: CfgNode,
        model: nn.Module,
        criterion: dict,
        grad_scaler,
        evaluator: Evaluator,
        device: torch.device,
        train_data_length: int,
        expe_name = "default",
        config_name = "default"
    ) -> None:
        
        self.cfg = cfg
        self.model = model
        self.device = device
        self.loss_scaler = NativeScaler()

        # solver related
        logger.info("\tSetting up the optimizer...")
        if 'vit' in self.cfg.MODEL.TYPE:
            self.optimizer = make_optimizer([self.model], cfg.SOLVER)
            self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        elif self.cfg.MODEL.TYPE == 'resnet':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg.SOLVER.BASE_LR, momentum=0.9, weight_decay=5e-4)
            if self.cfg.DATA.DATASET_TYPE == "landmark":
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [5, 10, 20], gamma=0.1, last_epoch=-1)
            elif self.cfg.DATA.DATASET_TYPE == "shop" or 'cub200' or 'sop':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.cfg.SOLVER.MILESTONES, gamma=0.1, last_epoch=-1)
        else:
            raise NotImplementedError
        self.criterion = criterion
        self.grad_scaler = grad_scaler
        self.feat_dim = model.module.feat_dim if self.cfg.NUM_GPUS > 1 else model.feat_dim
        self.embedding_dim = self.feat_dim if self.cfg.MODEL.PROJECTION_LAYERS < 0 else self.cfg.MODEL.PROJECTION_DIM
        self.xbm = XBM(
            memory_size=int(train_data_length * cfg.DATA.MEMORY_RATIO),
            embedding_dim=self.embedding_dim, device=device)
        self.checkpointer = Checkpointer(self.model, save_dir=cfg.OUTPUT_DIR, save_to_disk=True) 
        if cfg.MODEL.WEIGHT_PATH:
            # only use this for vtab in-domain experiments
            # checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, self.checkpointer.checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")
        
        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        if self.device == 0:
            loss_log_dir = "tensorboards/{:s}/{:s}/{:s}".format(self.cfg.DATA.NAME, expe_name, config_name)
            self.writer = SummaryWriter(log_dir=loss_log_dir)
            
    
    def train(self, train_loader, query_loader, gallery_loader):
        """
        Train a classifier using epoch
        """
        logger.info("max mem: {:.1f} GB ".format(gpu_mem_usage()))

        # save the model prompt if required before training
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        
        for epoch in range(1, total_epoch+1):
            if self.cfg.NUM_GPUS > 1 and self.cfg.DATA.RANDOM:
                train_loader.sampler.set_epoch(epoch)
            
            lr = self.optimizer.param_groups[0]['lr']
            logger.info("Training {} / {} epoch, with learning rate {}".format(epoch, total_epoch, lr))
            # Enable training mode
            self.model.train()

            if lr != 0:# skip this epoch
                self._train_epoch(train_loader, epoch, total_epoch)
            else:
                logger.info("The learning rate is zero, skip this epoch!")

             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            # eval at each epoch for single gpu training
            if epoch % self.cfg.EVAL.INTERVAL == 0 and (not self.cfg.EVAL.SKIP_EVAL):
                self.evaluator.update_iteration(epoch)
                if query_loader is not None and self.device == 0:# 只在主进程中测试
                    with torch.no_grad():
                        self.evaluator.evaluate(_model=self.model, _old_model=None,
                                                query_loader=query_loader, gallery_loader=gallery_loader, 
                                                log_writer=self.writer, epoch=epoch)

            if self.cfg.MODEL.SAVE_CKPT and self.device == 0:# 只在主进程中保存checkpoint
                self.checkpointer.save('{:s}_{:s}_{:s}_epoch_{:0>3d}'.format(self.cfg.DATA.NAME, self.cfg.MODEL.TRANSFER_TYPE, 
                                                                             self.cfg.MODEL.DATA.FEATURE, epoch))

    def _train_epoch(self, train_loader, epoch, n_epoch=0):
        """
        Training logic for an epoch
        """
        batch_time = AverageMeter('BatchTime', ':6.3f')
        data_time = AverageMeter('DataTime', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        len_dataloader = len(train_loader)
        progress = ProgressMeter(
            len_dataloader,
            [batch_time, data_time, losses],
            prefix=f"Epoch:[{epoch}/{n_epoch}]  ", logger=logger)

        end_time = time.time()

        for batch_idx, (images, labels, img_ids) in enumerate(train_loader):
            #print("self.device", self.device, img_ids)

            # measure data loading time
            data_time.update(time.time() - end_time)

            total_steps = epoch * len_dataloader + batch_idx
            
            # compute output
            with torch.cuda.amp.autocast(enabled=self.cfg.USE_AMP):
                images, labels, img_ids = images.to(self.device), labels.to(self.device), img_ids.to(self.device)
                
                feat, feat_k, cls_score = self.model(images.cuda(), return_feature=True)
                
                if self.cfg.LOSS.TYPE == "softmax":
                    loss = self.criterion['base'](cls_score, labels)
                
                elif self.cfg.LOSS.TYPE in ["arcface", "cosface"]:
                    if self.cfg.NUM_GPUS > 1:
                        cls_score = F.linear(F.normalize(feat), F.normalize(self.model.module.projection_cls.last_layer.weight))
                    else:
                        cls_score = F.linear(F.normalize(feat), F.normalize(self.model.projection_cls.last_layer.weight))
                    cls_score = large_margin_module(self.cfg.LOSS.TYPE, cls_score, labels,
                                                    s=self.cfg.LOSS.SCALE,
                                                    m=self.cfg.LOSS.MARGIN)
                    loss = self.criterion['base'](cls_score, labels)
                
                elif self.cfg.LOSS.TYPE in ["contra", "supcontra"]:
                    feat_norm = F.normalize(feat, dim=1, p=2.0)
                    feat_k_norm = F.normalize(feat_k, dim=1, p=2.0) if feat_k is not None else feat_norm

                    contrastive_loss = self.criterion["base"](feat_norm, labels)
                    if self.cfg.LOSS.USE_XBM:
                        self.xbm.enqueue_dequeue(feat_k_norm.detach(), labels.detach(), img_ids.detach()) 
                        xbm_feat_norm, xbm_labels, xbm_img_ids = self.xbm.get()
                        if self.cfg.LOSS.TYPE == "contra":
                            xbm_contrastive_loss = self.criterion["base"](feat_norm, labels, ref_emb=xbm_feat_norm, ref_labels=xbm_labels)
                        elif self.cfg.LOSS.TYPE == "supcontra":
                            xbm_contrastive_loss = self.criterion["base"](feat_norm, labels, img_ids=img_ids, 
                                      ref_features=xbm_feat_norm, ref_labels=xbm_labels, ref_img_ids=xbm_img_ids)
                    else:
                        xbm_contrastive_loss = torch.tensor(0.0, device=self.device)
                    loss = contrastive_loss + xbm_contrastive_loss
            
            if self.device==0:
                self.writer.add_scalar("Loss", losses.avg, total_steps)
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            losses.update(loss.item(), images.size(0))
            # grad_scaler can handle the case that use_amp=False indicated in the official pytorch doc
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % self.cfg.SOLVER.LOG_EVERY_N == 0:
                #progress.display(batch_idx, suffix=f"\tlr:{self.scheduler.get_lr()[0]:.6f}")
                if self.cfg.SOLVER.BACKBONE_MULTIPLIER != 1.:
                    progress.display(batch_idx, suffix=f"\tgroup0 lr:{self.optimizer.param_groups[0]['lr']:.6f}" + f"\tgroup1 lr:{self.optimizer.param_groups[2]['lr']:.6f}")
                else:
                    progress.display(batch_idx, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.6f}")
            if self.device==0:
                #self.writer.add_scalar("lr", self.scheduler.get_lr()[0], total_steps)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], total_steps)

            # if batch_idx == 2:
            #     break


def count_parameters_excluding(model, exclude_names):
    return sum(p.numel() for name, p in model.named_parameters() 
               if p.requires_grad and not any(exclude in name for exclude in exclude_names))
    
def count_parameters_including(model, include_names):
    return sum(p.numel() for name, p in model.named_parameters() 
               if p.requires_grad and any(include in name for include in include_names))
    
class SubModelTrainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self, 
        cfg: CfgNode,
        model: nn.Module,
        criterion: dict,
        grad_scaler,
        evaluator: Evaluator,
        device: torch.device,
        train_data_length: int,
        K: int=0,
        expe_name = "default",
        config_name = "default"
    ) -> None:
        
        self.cfg = cfg
        self.model = model
        self.device = device
        self.loss_scaler = NativeScaler()
        self.iter_per_epoch = train_data_length
        self.total_iteraions = self.iter_per_epoch * self.cfg.SOLVER.TOTAL_EPOCH
        # solver related
        logger.info("\tSetting up the optimizer...")
        self.model_list = [self.model]
        self.multi_independent_model_list = None
        if self.cfg.COMP_LOSS.TYPE in ['multi_independent']:
            self.multi_independent_model_list = build_sub_models(cfg=self.cfg, sparsity=self.cfg.SNET.WIDTH_MULT_LIST,model=self.model)
            self.model_list.extend(self.multi_independent_model_list)
            
        if self.cfg.MODEL.TYPE in ['vit','our_vit','timm-vit','our_timm_vit','convnext','swin-vit']  :
            self.optimizer = make_optimizer(self.model_list, cfg.SOLVER)
            self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER,iter_per_epoch=self.iter_per_epoch,total_iter=self.total_iteraions)
        elif self.cfg.MODEL.TYPE in ['resnet', 'our_mobilenet','resnet_group','resnext','resnext_switch','mobile_switch']:
            self.optimizer = make_optimizer(self.model_list, cfg.SOLVER)
            if self.cfg.DATA.DATASET_TYPE == "landmark":
                self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER,iter_per_epoch=self.iter_per_epoch,total_iter=self.total_iteraions)
            elif self.cfg.DATA.DATASET_TYPE == "shop" or 'cub200' or 'sop':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.cfg.SOLVER.MILESTONES, gamma=0.1, last_epoch=-1)
        else:
            raise NotImplementedError
        self.criterion = criterion
        self.grad_scaler = grad_scaler
        self.backbone_feat_dim = model.module.feat_dim if self.cfg.NUM_GPUS > 1 else model.feat_dim

        self.embedding_dim = self.backbone_feat_dim if self.cfg.MODEL.PROJECTION_LAYERS < 0 \
                             else self.cfg.MODEL.PROJECTION_DIM  

        self.xbm = XBM(
            memory_size=int(train_data_length * cfg.DATA.MEMORY_RATIO),
            embedding_dim=self.embedding_dim, device=device)

        self.checkpointer = Checkpointer(self.model,optimizer=self.optimizer,scheduler=self.scheduler,save_dir=cfg.OUTPUT_DIR, save_to_disk=True)
        if self.cfg.COMP_LOSS.TYPE in ['SFSC']:
            from .PCGrad import PCGrad
            self.pcg = PCGrad(cfg, self.model, self.optimizer,reduction='sum',FP16_ENABLED=False, Conflict='ALL')
            
        if self.cfg.MODEL.WEIGHT_PATH:
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, self.checkpointer.checkpointables)   
            logger.info(f"New model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        if self.cfg.COMP_LOSS.TYPE == 'independent':
            for name, param in self.model.named_parameters():
                if 'w_m' in name:
                    param.requires_grad = False
                    
        if self.cfg.COMP_LOSS.TYPE in ['BCT']:
            from ..models.build_model import build_sub_models
            _cfg = self.cfg.clone()
            _cfg._immutable(False)
            _cfg.SNET.WIDTH_MULT_LIST = [0.8944,0.7746,0.6325,0.4472,1.0]
            self.parent_model,_ = build_sub_models(_cfg,sparsity=0)
            print(self.parent_model)
            try:
                _dict = torch.load(self.cfg.SUB_MODEL.PARENT_MODEL_WEIGHT)['model']
                if self.cfg.NUM_GPUS > 1:
                    new_dict = {}
                    for key,value in _dict.items():
                        new_key = 'module.'+ key
                        new_dict[new_key] = value
                    missing_keys, unexpected_keys = self.parent_model.load_state_dict(new_dict, strict=False)
                else:
                    missing_keys, unexpected_keys = self.parent_model.load_state_dict(_dict, strict=False)
                # Display missing and unexpected keys
                if missing_keys:
                    print("Missing keys (parameters not found in the state dict):")
                    for key in missing_keys:
                        print(f"  - {key}")

                if unexpected_keys:
                    print("Unexpected keys (parameters in state dict that are not in the model):")
                    for key in unexpected_keys:
                        print(f"  - {key}")

            except Exception as e:
                print(f"Failed to load model weights: {e}")

            for n,p in self.parent_model.named_parameters():
                p.requires_grad = False
                


        self.evaluator = evaluator
        if self.device == 0:
            loss_log_dir = cfg.OUTPUT_DIR
            self.writer = SummaryWriter(log_dir=loss_log_dir)

    def train(self, train_loader, query_loader, gallery_loader, expe_name = "default"):
        """
        Train a classifier using epoch
        """

        logger.info("max mem: {:.1f} GB ".format(gpu_mem_usage()))
        # """
        if self.cfg.COMP_LOSS.ONLY_TEST:
            parent_sparsity = self.cfg.SUB_MODEL.PARENT_SPARSITY
            sub_model_list = None
            sub_model_list = build_sub_models(cfg=self.cfg,sparsity=self.cfg.SUB_MODEL.SPARSITY if not self.cfg.SUB_MODEL.RANDOM_SPARSITY else [torch.rand(1)],model=self.model)

            if self.cfg.COMP_LOSS.TYPE in ['BCT']:
                self.model.eval()
                self.parent_model.eval()
                with torch.no_grad():
                    # sub_model_list = build_sub_models(cfg=self.cfg,sparsity=self.cfg.SUB_MODEL.SPARSITY if not self.cfg.SUB_MODEL.RANDOM_SPARSITY else [torch.rand(1)],model=self.parent_model)
                    sub_model_list = [self.parent_model for i in [0.8944,0.7746,0.6325,0.4472,1.0]]
                    if query_loader is not None and self.device == 0:
                        self.evaluator.evaluate(parent_model=self.model,query_loader=query_loader, gallery_loader=gallery_loader, log_writer=self.writer, epoch=1,sub_model_list=sub_model_list,parent_sparsity=self.cfg.SUB_MODEL.PARENT_SPARSITY)
                exit()
                                                                                       
            if self.cfg.SUB_MODEL.USE_SWITCHNET==True:    
                if self.cfg.SUB_MODEL.ADAPTIVE_BN:
                    from torch.utils.data.sampler import RandomSampler
                    from ..models.subnetworks.slimmable_ops import SwitchableBatchNorm2d,SwitchableBatchNorm1d
                    
                    original_dataset_size = len(train_loader.dataset)
                    if self.cfg.DATA.NAME == 'retrieval-inshop':
                        new_dataset_size = original_dataset_size // 1 # inshop
                    else:
                        new_dataset_size = original_dataset_size // 30 # gldv2
                    sampler = RandomSampler(train_loader.dataset, replacement=False, num_samples=new_dataset_size)
                    new_train_loader = torch.utils.data.DataLoader(
                        train_loader.dataset,
                        batch_size=train_loader.batch_size,
                        sampler=sampler,
                        num_workers=train_loader.num_workers
                    )
                    total_batches = len(new_train_loader)
                    self.model.train()
                    model_params = [param for param in self.model.parameters() if param.requires_grad]
                    for param in model_params:
                        param.requires_grad = False
                    for module in self.model.modules():
                        if isinstance(module, (SwitchableBatchNorm2d,SwitchableBatchNorm1d)):
                            for bn in module.bn:
                                # print(bn.running_mean)
                                bn.reset_running_stats()
                            for param in module.parameters():
                                param.requires_grad = False            
                    for batch_idx, (images, labels, img_ids) in enumerate(new_train_loader):
                        self.model.apply(lambda m: setattr(m, 'width_mult', parent_sparsity))
                        _, _, _ = self.model(images.cuda(),return_feature=True)
                        if not self.cfg.SUB_MODEL.ONLY_TEST_PARENT:
                            for idx,sparsity in enumerate(self.cfg.SNET.WIDTH_MULT_LIST):
                                if sparsity != parent_sparsity:
                                    self.model.apply(lambda m: setattr(m, 'width_mult', sparsity))
                                    _, _, _ = self.model(images.cuda(), return_feature=True)
                        # logger.info('parent model adaptive finish')
                        if (batch_idx+1) % 20 == 0 and self.device == 0:
                            logger.info(f'parent model ,Batch {batch_idx+1}/{total_batches}')                                                              
                self.model.eval()
                if query_loader is not None and self.device == 0:# 
                    with torch.no_grad():
                        self.evaluator.evaluate(parent_model=self.model,query_loader=query_loader, gallery_loader=gallery_loader, log_writer=self.writer, epoch=1,sub_model_list=sub_model_list,parent_sparsity=parent_sparsity)
                exit()                               
            sub_model_list = build_sub_models(cfg=self.cfg,sparsity=self.cfg.SUB_MODEL.SPARSITY if not self.cfg.SUB_MODEL.RANDOM_SPARSITY else [torch.rand(1)],model=self.model)
            if self.cfg.SUB_MODEL.ADAPTIVE_BN:
                from torch.utils.data.sampler import RandomSampler
                sub_model_list = build_sub_models(cfg=self.cfg,sparsity=self.cfg.SUB_MODEL.SPARSITY if not self.cfg.SUB_MODEL.RANDOM_SPARSITY else [torch.rand(1)],model=self.model)
                
                original_dataset_size = len(train_loader.dataset)
                if self.cfg.DATA.NAME == 'retrieval-inshop':
                    new_dataset_size = original_dataset_size // 1 # inshop
                else:
                    new_dataset_size = original_dataset_size // 30 # gldv2
                sampler = RandomSampler(train_loader.dataset, replacement=False, num_samples=new_dataset_size)
                new_train_loader = torch.utils.data.DataLoader(
                    train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    sampler=sampler,
                    num_workers=train_loader.num_workers
                )
                total_batches = len(new_train_loader)
                self.model.train()
                model_params = [param for param in self.model.parameters() if param.requires_grad]

                for param in model_params:
                    param.requires_grad = False

                for module in self.model.modules():
                    if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                        module.reset_running_stats()
                        for param in module.parameters():
                            param.requires_grad = False     
                if not self.cfg.SUB_MODEL.ONLY_TEST_PARENT:                                                
                    for idx,sub_model in enumerate(sub_model_list):
                        sparsity = self.cfg.SUB_MODEL.SPARSITY[idx]
                        sub_model.train()
                        model_params = [param for param in sub_model.parameters() if param.requires_grad]

                        for param in model_params:
                            param.requires_grad = False

                        for module in sub_model.modules():
                            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                                module.reset_running_stats()
                                for param in module.parameters():
                                    param.requires_grad = False          

                for batch_idx, (images, labels, img_ids) in enumerate(new_train_loader):
                    _, _, _ = self.model(images.cuda(), sparsity=self.cfg.SUB_MODEL.PARENT_SPARSITY, mask=None, return_feature=True)
                    if not self.cfg.SUB_MODEL.ONLY_TEST_PARENT:                                                
                        for idx,sub_model in enumerate(sub_model_list):
                            sparsity = self.cfg.SUB_MODEL.SPARSITY[idx]  
                            _, _, _ = sub_model(images.cuda(), sparsity=sparsity, mask=None, return_feature=True)
                    # logger.info('parent model adaptive finish')
                    if (batch_idx+1) % 20 == 0 and self.device == 0:
                        logger.info(f'parent model ,Batch {batch_idx+1}/{total_batches}')
                logger.info('finish adaptive bn train')
                
            self.model.eval()
            if query_loader is not None and self.device == 0:# 只在主进程中测试
                with torch.no_grad():
                    self.evaluator.evaluate(parent_model=self.model,query_loader=query_loader, gallery_loader=gallery_loader, log_writer=self.writer, epoch=1,sub_model_list=sub_model_list,parent_sparsity=self.cfg.SUB_MODEL.PARENT_SPARSITY)
            exit()
        # """
        # save the model prompt if required before training
        self.model.eval()
        
        # setup training epoch params
        if self.cfg.MODEL.WEIGHT_PATH:
            epoch_str = self.cfg.MODEL.WEIGHT_PATH.split("epoch_")[-1].split(".")[0]
            start_epoch = int(epoch_str) + 1
        else:
            start_epoch = 1
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(start_epoch, total_epoch+1):
            if self.cfg.SUB_MODEL.FREEZEW_M:              
                for name, param in self.model.named_parameters():
                    if 'w_m' in name:
                        param.requires_grad = False
            lr = self.optimizer.param_groups[0]['lr']
            logger.info("Training {} / {} epoch, with learning rate {}".format(epoch, total_epoch, lr))
            self.model.train()
            self._back_comp_train_epoch(train_loader, epoch, total_epoch)
        
            if self.cfg.SOLVER.OPTIMIZER == 'SGD':
                self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            if self.cfg.MODEL.SAVE_CKPT and self.device == 0 and epoch % self.cfg.EVAL.INTERVAL == 0:# 只在主进程中保存checkpoint
                if self.cfg.COMP_LOSS.TYPE in ['multi_independent']:
                    for idx,sparsity in enumerate(self.cfg.SNET.WIDTH_MULT_LIST):
                        if sparsity != 1.0:
                            _checkpointer = Checkpointer(self.multi_independent_model_list[idx],optimizer=self.optimizer,scheduler=self.scheduler,save_dir=self.cfg.OUTPUT_DIR, save_to_disk=True)
                            _checkpointer.save('{:s}_{:s}_{:s}_epoch_{:0>3d}_sub_{:s}'.format(self.cfg.DATA.NAME, self.cfg.MODEL.TRANSFER_TYPE, self.cfg.MODEL.DATA.FEATURE, epoch,str(sparsity)))
                            
                self.checkpointer.save('{:s}_{:s}_{:s}_epoch_{:0>3d}'.format(self.cfg.DATA.NAME, self.cfg.MODEL.TRANSFER_TYPE, 
                                                                             self.cfg.MODEL.DATA.FEATURE, epoch))
                
            # eval at each epoch for single gpu training
            if epoch % self.cfg.EVAL.INTERVAL == 0 and (not self.cfg.EVAL.SKIP_EVAL):
                self.evaluator.update_iteration(epoch)
                if query_loader is not None and self.device == 0:# 只在主进程中测试
                    with torch.no_grad():
                        self.evaluator.evaluate(parent_model=self.model,query_loader=query_loader, gallery_loader=gallery_loader, log_writer=self.writer, epoch=epoch,sub_model_list=self.multi_independent_model_list,parent_sparsity=self.cfg.SUB_MODEL.PARENT_SPARSITY)
                        # if self.cfg.COMP_LOSS.TYPE in ['multi_independent']:
                        #     self.evaluator.evaluate(parent_model=self.model,query_loader=query_loader, gallery_loader=gallery_loader, log_writer=self.writer, epoch=epoch,)

            # self.evaluator.update_iteration(epoch)
        if self.device == 0:
            self.model.eval()
            self.evaluator.evaluate(parent_model=self.model,query_loader=query_loader, gallery_loader=gallery_loader, log_writer=self.writer, epoch=epoch,parent_sparsity=self.cfg.SUB_MODEL.PARENT_SPARSITY)


    def compare_two_tensors(self,tensor1,tensor2):
        flat_tensor1 = tensor1.flatten()
        flat_tensor2 = tensor2.flatten()
        sorted_indices1 = torch.argsort(flat_tensor1)
        sorted_indices2 = torch.argsort(flat_tensor2)
        # 比较排序后的索引，计算不相等元素的数量
        num_changed = torch.sum(sorted_indices1 != sorted_indices2).item()
        # 打印结果
        print("Number of changed indices:", num_changed)

    def fix_bn(self,m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def free_bn(self,m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    def _back_comp_train_epoch(self, train_loader, epoch, n_epoch=0):
        batch_time = AverageMeter('BatchTime', ':6.3f')
        data_time = AverageMeter('DataTime', ':6.3f')
        losses_cls = AverageMeter('Cls Loss', ':.4f')

        losses_all = AverageMeter('Total Loss', ':.4f')
        meter_list = [batch_time, data_time, losses_all, losses_cls]
        submodel_loss = {}
        sub_loss = None

        if self.cfg.COMP_LOSS.TYPE in ['SFSC','asymmetric','mul_score_map','proj_with_cosine_sim','multi_independent','BCT_S','BCT','proj_with_cosine_sim_switchnet']:# compatible loss
            # losses_supcontrast_comp = AverageMeter('Comp supcontrastive Loss', ':.4f')
            # meter_list.append(losses_supcontrast_comp)
            for idx,sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                submodel_loss[idx] = AverageMeter('Submodel ' + str(sparsity) + ' Loss', ':.4f')
                meter_list.append(submodel_loss[idx])

        
        if self.cfg.SUB_MODEL.BCT_S:# compatible loss
            losses_sub_model_cls = AverageMeter('submodel cls Loss', ':.4f')
            meter_list.append(losses_sub_model_cls) 

        len_dataloader = len(train_loader)
        progress = ProgressMeter(
            len_dataloader,
            meter_list,
            prefix=f"Epoch:[{epoch}/{n_epoch}]  ", logger=logger)

        end_time = time.time()

        if self.cfg.UPGRADE_LOSS.TYPE == 'others':
            for name, param in self.model.named_parameters():
                if 'projection_cls' in name:# used to be projection_cls
                    param.requires_grad = False
        for batch_idx, (images, labels, img_ids) in enumerate(train_loader):
            # img, pid, camid, trackid,img_path
            # dist.barrier()
            if self.device == 0:
                self.writer.add_scalar("y=x",batch_idx, batch_idx)


            data_time.update(time.time() - end_time)
            total_steps = epoch * len_dataloader + batch_idx
            with torch.cuda.amp.autocast(enabled=self.cfg.USE_AMP):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.cfg.SOLVER.TYPE == 'submodel_compatible':
                    for name,param in self.model.named_parameters():
                        if 'projector' in name or 'projection_cls' in name:
                            param.requires_grad = True
                    if self.cfg.SUB_MODEL.USE_SWITCHNET:
                        if self.cfg.COMP_LOSS.TYPE in ['independent','BCT']:
                            self.model.apply(lambda m: setattr(m, 'width_mult', self.cfg.SNET.WIDTH_MULT_LIST[0]))
                        else:
                            self.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                        if not self.cfg.COMP_LOSS.TYPE in ['BCT','proj_with_cosine_sim','mul_score_map','proj_with_cosine_sim_switchnet','SFSC']: 
                            feat, _, cls_score = self.model(images.cuda(), return_feature=True)
                    else:
                        feat, feat_k, cls_score = self.model(images.cuda(), sparsity=0, mask=None, return_feature=True)

                    if self.cfg.LOSS.TYPE == 'supcontra':
                        feat_norm = F.normalize(feat, dim=1, p=2.0)
                        # feat_norm = feat
                        loss = self.criterion['base'](feat_norm, labels)
                        # print(loss)
                    # elif self.cfg.LOSS.TYPE = 'uncertainty':
                    #     loss = 
                    else:
                        if self.cfg.SUB_MODEL.ADD_PARENT_HARD_SCALE:
                            loss_criterion = torch.nn.CrossEntropyLoss(reduction='none')
                            loss = loss_criterion(cls_score, labels)
                            loss_p = loss.detach()
                            mean = loss_p.mean()
                            # std = loss.std()
                            loss_mean = (loss_p - mean) / mean
                            hard_sacle_parent = 1 + loss_mean
                            loss = torch.mean(loss * hard_sacle_parent)
                        else:
                            if not self.cfg.COMP_LOSS.TYPE in ['BCT','proj_with_cosine_sim','mul_score_map','proj_with_cosine_sim_switchnet','SFSC']:
                                loss = self.criterion['base'](cls_score, labels)
                            else:
                                loss = 0
                            

                if self.cfg.COMP_LOSS.TYPE in ['multi_independent']:
                    sub_loss = []
                    compatible_loss = 0.0
                    try:
                        model_cls = self.model.module.projection_cls
                    except Exception as e:
                        model_cls = self.model.projection_cls
                    # for i in model_cls.parameters():
                    #     i.requires_grad = False
                    for idx, _sub_independent_model in enumerate(self.multi_independent_model_list):
                        sparsity = self.cfg.SNET.WIDTH_MULT_LIST[idx]
                        _sub_independent_model.train()
                        try:
                            _sub_independent_model.module.apply(lambda m: setattr(m, 'width_mult', sparsity))
                        except Exception as e:
                            _sub_independent_model.apply(lambda m: setattr(m, 'width_mult', sparsity))
                        feat_sub, _, _ = _sub_independent_model(images.cuda(), return_feature=True)                           
                        cls_score_sub = model_cls(feat_sub)
                        loss_sub = self.criterion['base'](cls_score_sub, labels)
                        compatible_loss += loss_sub
                        sub_loss.append(loss_sub)
                    # for i in model_cls.parameters():
                    #     i.requires_grad = True 
                                
                elif self.cfg.COMP_LOSS.TYPE in ['asymmetric']:
                    assert self.cfg.BN_TRACK == False
                    sub_loss = []
                    compatible_loss = 0.0
                    if self.cfg.SUB_MODEL.USE_SWITCHNET:
                        for idx,sparsity in enumerate(self.cfg.SNET.WIDTH_MULT_LIST):
                            if sparsity != 1.0:
                                self.model.apply(lambda m: setattr(m, 'width_mult', sparsity))
                                feat_sub, _, cls_score_sub = self.model(images.cuda(), return_feature=True)
                                assymmetric_loss = self.criterion['back_comp'](feat_sub,feat,labels,margin=0.3,norm_feat=True,hard_mining=True)
                                sub_loss.append(assymmetric_loss)
                                compatible_loss += assymmetric_loss
                    else:           
                        for idx,sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                            feat_sub, _, cls_score_sub = self.model(images.cuda(), sparsity=sparsity, mask=None,return_feature=True)
                            assymmetric_loss = self.criterion['back_comp'](feat_sub,feat,labels,margin=0.3,norm_feat=True,hard_mining=True)
                            sub_loss.append(assymmetric_loss)
                            compatible_loss += assymmetric_loss
                                
                elif self.cfg.COMP_LOSS.TYPE == 'independent':
                    compatible_loss = 0.0

                elif self.cfg.COMP_LOSS.TYPE in ['mul_score_map']:
                    compatible_loss = 0.0
                    gradients = {}
                    sub_loss = []
                    gradients['parent'] = {}
                    for name,param in self.model.named_parameters():
                        if 'projector' in name or 'projection_cls' in name:
                            param.requires_grad = False
                            # print(param.grad)

                    for idx,sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                        compatible_loss_sub = 0.0
                        gradients[sparsity] = {}
                        feat_sub, _, cls_score_sub = self.model(images.cuda(), sparsity=sparsity, mask=None,return_feature=True)
                        if self.cfg.SUB_MODEL.SUB_MODEL_LOSS_TYPE =='cls':
                            if self.cfg.SUB_MODEL.ADD_KL_SCALE:                            
                                loss_criterion = torch.nn.CrossEntropyLoss(reduction='none')
                                compatible_loss += loss_criterion(cls_score_sub, labels)
                                if self.cfg.SUB_MODEL.ADD_HARD_SCALE:
                                    compatible_loss_sub = compatible_loss.detach()
                                    mean = compatible_loss_sub.mean()
                                    # std = compatible_loss.std()
                                    compatible_loss_mean = (compatible_loss_sub - mean) / mean
                                    hard_sacle = 1 + compatible_loss_mean
                                cls_score_norm = F.softmax(cls_score, dim=-1)
                                cls_score_sub_norm = F.softmax(cls_score_sub, dim=-1)
                                p = cls_score_norm.detach() + 1e-10
                                q = cls_score_sub_norm.detach() + 1e-10
                                kl_divergence = torch.sum(p * torch.log(p / q), dim=-1)
                                mean_kl = torch.mean(kl_divergence)
                                kl_divergence = (kl_divergence - mean_kl)
                                # noise = torch.randn_like(compatible_loss) * mean_kl
                                if self.cfg.SUB_MODEL.CHANGE_KL_SCALE:
                                    if epoch > self.cfg.SUB_MODEL.SCALE_START_EPCOH:
                                        compatible_loss = torch.mean(compatible_loss * (1 + kl_divergence) * hard_sacle)
                                    else:
                                        compatible_loss = torch.mean(compatible_loss * (1 - kl_divergence)) # 先学习简单样本
                                else:
                                    compatible_loss = torch.mean(compatible_loss * (1 + kl_divergence))
                            else:
                                compatible_loss_sub += self.criterion['base'](cls_score_sub, labels)
                                sub_loss.append(compatible_loss_sub)
                                compatible_loss += compatible_loss_sub
                        self.optimizer.zero_grad()
                        self.grad_scaler.scale(compatible_loss_sub).backward(retain_graph=False)                            
                        if self.cfg.SUB_MODEL.GRAD_PROJ:
                            for name, module in self.model.named_modules():
                                if 'conv' in name and 'w_m' not in name:
                                    gradients[sparsity][name+'.weight'] = module.weight.grad.clone()
                                    if module.bias is not None:
                                        gradients['parent'][name+'.bias'] = module.bias.grad.clone()
                    # compatible_loss /= len(self.cfg.SUB_MODEL.SPARSITY)   
                      
                    for name,param in self.model.named_parameters():
                        if 'projector' in name or 'projection_cls' in name:
                            param.requires_grad = True    
                                                           
                    if self.cfg.SUB_MODEL.USE_SWITCHNET:
                        if self.cfg.COMP_LOSS.TYPE in ['independent','BCT']:
                            self.model.apply(lambda m: setattr(m, 'width_mult', self.cfg.SNET.WIDTH_MULT_LIST[0]))
                        else:
                            self.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                        feat, _, cls_score = self.model(images.cuda(), return_feature=True)
                    else:
                        feat, _, cls_score = self.model(images.cuda(), sparsity=0, mask=None, return_feature=True)
                        
                    loss = self.criterion['base'](cls_score, labels)
                                                                                                   
                    if self.cfg.SUB_MODEL.GRAD_PROJ:
                        self.optimizer.zero_grad()
                        self.grad_scaler.scale(loss).backward(retain_graph=False)

                        for name, module in self.model.named_modules():
                                if isinstance(module,(SubnetConv2d,SubnetLinear)):
                                    gradients['parent'][name+'.weight'] = module.weight.grad.clone()
                                    if module.bias is not None:
                                        gradients['parent'][name+'.bias'] = module.bias.grad.clone()


    
                    for name, module in self.model.named_modules():
                        proj_condition = isinstance(module,(SubnetConv2d,SubnetLinear))
                        if proj_condition:
                            for _name,param in module.named_parameters():
                                full_name = name + '.' + _name
                                if 'w_m' in full_name or 'b_m' in full_name:
                                    continue
                                g_list = []
                                g_ret_list = []
                                g_list.append(gradients['parent'][full_name])
                                for idx, sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                                    if self.cfg.SUB_MODEL.USE_SWITCHNET:
                                        sparsity = self.cfg.SNET.WIDTH_MULT_LIST[idx]
                                    g_list.append(gradients[sparsity][full_name])

                                g_list_ori = [i.detach() for i in g_list]
                                if isinstance(module,(SubnetConv2d,SlimmableConv2d)) and len(param.shape)==4 and self.cfg.SUB_MODEL.PROJ_BY_KERNEL:
                                    g_shuffle = random.sample(g_list, len(g_list))
                                    for g_i in g_list:
                                        g_tmp = g_i.clone()
                                        for g_j in g_shuffle:
                                            g_i_flatten = g_i.view(g_i.shape[0], -1)
                                            g_j_flatten = g_j.view(g_i.shape[0], -1)
                                            vec_dot_product = torch.sum(g_i_flatten * g_j_flatten, dim=1)
                                            negative_indices = torch.where(vec_dot_product < 0)[0].tolist()
                                            if len(negative_indices) >0:
                                                tmp_i = g_i[negative_indices,:,:,:]
                                                tmp_j = g_j[negative_indices,:,:,:]
                                                tmp_j_flatten = tmp_j.view(len(negative_indices), -1)
                                                scale = vec_dot_product[negative_indices] / (torch.sum(tmp_j_flatten * tmp_j_flatten, dim=1)+1e-8)
                                                scale_expanded = scale.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(tmp_i)
                                                g_i_proj = tmp_i - scale_expanded * tmp_j
                                                g_tmp[negative_indices,:,:,:] = g_i_proj
                                        g_ret_list.append(g_tmp.detach())
                                else:
                                    for g_i in g_list:
                                        g_tmp = g_i.clone()
                                        g_shuffle = random.sample(g_list, len(g_list))
                                        for g_j in g_shuffle: 
                                            d = torch.dot(g_i.flatten(), g_j.flatten())  
                                            if d < 0:
                                                g_tmp_i = g_i
                                                g_tmp_j = g_j
                                                g_tmp = (g_tmp_i - d / (torch.dot(g_tmp_j.view(-1), g_tmp_j.view(-1))+1e-8) * g_tmp_j)
                                        g_ret_list.append(g_tmp.detach())

                                total_len = len(self.cfg.SUB_MODEL.SPARSITY)+1
                                cos_sim = []
                                for i in range(total_len):
                                    flat_tensor1 = g_ret_list[i].view(-1)
                                    flat_tensor2 = g_list_ori[i].view(-1)                                            
                                    if self.cfg.SUB_MODEL.COS_SIM.USE_DOT:
                                        dot_sim = torch.dot(flat_tensor1.view(-1), flat_tensor2.view(-1))
                                        cos_sim.append((dot_sim+1)/2)
                                    else:
                                        cos_sim_i = torch.nn.functional.cosine_similarity(flat_tensor1.unsqueeze(0), flat_tensor2.unsqueeze(0))
                                        cos_sim.append((cos_sim_i+1)/2)
                                g_tensor = torch.stack(g_ret_list,dim=0)
                                cos_sim = torch.tensor(cos_sim, dtype=torch.float32).to(self.device)
                                cos_sim = cos_sim ** self.cfg.SUB_MODEL.COS_SIM.EXP
                                cos_sim = cos_sim / (cos_sim.sum()+1e-8)
                                g_ratio = cos_sim
                                # print(cos_sim)
                                # g_ratio = cos_sim.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                                for _ in range(len(g_tensor.shape) - 1):
                                    g_ratio = g_ratio.unsqueeze(1)
                                g_ratio = g_ratio.expand_as(g_tensor)
                                g_ret = torch.sum(g_tensor * g_ratio, dim=0) * total_len
                                param.grad = g_ret                    


                elif self.cfg.COMP_LOSS.TYPE in ['proj_with_cosine_sim']:
                    compatible_loss_all = 0.0
                    count = 0
                    sub_loss = []
                    gradients = {}
                    if not self.cfg.SUB_MODEL.USE_SWITCHNET:
                        sub_model_list = build_sub_models(cfg=self.cfg,sparsity=self.cfg.SUB_MODEL.SPARSITY if not self.cfg.SUB_MODEL.RANDOM_SPARSITY else [torch.rand(1)],model=self.model)
                    if self.cfg.SUB_MODEL.GRAD_PROJ:
                        gradients['parent'] = {}
                    ratio = [1] + [1 for i in self.cfg.SUB_MODEL.SPARSITY]
                                                                
                    # for name,param in self.model.named_parameters():
                    #     if 'projection_cls.last_layer' in name:
                    #         param.requires_grad = False
                            # print(param)
                                                                                            
                    for idx,sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                        if not self.cfg.SUB_MODEL.USE_SWITCHNET:
                            _model = sub_model_list[idx]
                        else:
                            sparsity = self.cfg.SNET.WIDTH_MULT_LIST[idx]
                            self.model.apply(lambda m: setattr(m, 'width_mult', sparsity))
                            _model = self.model
                        if self.cfg.SUB_MODEL.GRAD_PROJ:
                            compatible_loss = 0.0
                            gradients[sparsity] = {}
                        if not self.cfg.SUB_MODEL.USE_SWITCHNET:
                            feat_sub, _, cls_score_sub = _model(images.cuda(), sparsity=sparsity, mask=None,return_feature=True)
                        else:
                            feat_sub, _, cls_score_sub = _model(images.cuda(), return_feature=True)
                        if self.cfg.SUB_MODEL.SUB_MODEL_LOSS_TYPE =='cls':
                            if self.cfg.SUB_MODEL.ADD_KL_SCALE:                            
                                loss_criterion = torch.nn.CrossEntropyLoss(reduction='none')
                                compatible_loss += loss_criterion(cls_score_sub, labels)                               
                                if self.cfg.SUB_MODEL.ADD_HARD_SCALE:
                                    compatible_loss_sub = compatible_loss.detach()
                                    mean = compatible_loss_sub.mean()
                                    # std = compatible_loss.std()
                                    compatible_loss_mean = (compatible_loss_sub - mean) / mean
                                    hard_sacle = 1 + compatible_loss_mean
                                cls_score_norm = F.softmax(cls_score, dim=-1)
                                cls_score_sub_norm = F.softmax(cls_score_sub, dim=-1)
                                p = cls_score_norm.detach() + 1e-10
                                q = cls_score_sub_norm.detach() + 1e-10
                                kl_divergence = torch.sum(p * torch.log(p / q), dim=-1)
                                mean_kl = torch.mean(kl_divergence)
                                kl_divergence = (kl_divergence - mean_kl)
                                # noise = torch.randn_like(compatible_loss) * mean_kl
                                if self.cfg.SUB_MODEL.CHANGE_KL_SCALE:
                                    if epoch > self.cfg.SUB_MODEL.SCALE_START_EPCOH:
                                        compatible_loss = torch.mean(compatible_loss * (1 + kl_divergence) * hard_sacle)
                                    else:
                                        compatible_loss = torch.mean(compatible_loss * (1 - kl_divergence)) # 先学习简单样本
                                else:
                                    compatible_loss = torch.mean(compatible_loss * (1 + kl_divergence))
                            else:
                                _sub_loss = self.criterion['base'](cls_score_sub, labels)
                                sub_loss.append(_sub_loss)
                                if self.cfg.SUB_MODEL.ADD_LOSS_SCALE:
                                    loss_scale = (1 - float(sparsity)) if not self.cfg.SUB_MODEL.USE_SWITCHNET else (1-float(sparsity)**2)
                                    _sub_loss *= loss_scale
                                compatible_loss += _sub_loss
                                compatible_loss_all += compatible_loss
                        elif self.cfg.SUB_MODEL.SUB_MODEL_LOSS_TYPE =='comp':
                            feat_sub_norm = F.normalize(feat_sub, dim=1, p=2.0)
                            # feat_norm = F.normalize(feat, dim=1, p=2.0)
                            _sub_loss = self.criterion["back_comp"](feat_sub_norm, labels, feat_norm.detach(), labels)
                            sub_loss.append(_sub_loss)
                            compatible_loss += _sub_loss
                            compatible_loss_all += compatible_loss                        
                                 
                        self.optimizer.zero_grad()
                        self.grad_scaler.scale(compatible_loss).backward(retain_graph=False)
                        torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=100)                            
                        if self.cfg.SUB_MODEL.GRAD_PROJ:
                            for name, module in _model.named_modules():
                                if isinstance(module,(SubnetConv2d,SubnetLinear,SlimmableConv2d,SlimmableLinear)):
                                    gradients[sparsity][name+'.weight'] = module.weight.grad.clone()
                                    if module.bias is not None:
                                        gradients[sparsity][name+'.bias'] = module.bias.grad.clone()
                                    if not self.cfg.SUB_MODEL.FREEZEW_M and self.cfg.SUB_MODEL.PROJ_W_M and not self.cfg.SUB_MODEL.USE_SWITCHNET:
                                        gradients[sparsity][name+'.w_m'] = module.w_m.grad.clone() * module.weight_mask
                                        if module.bias is not None:
                                            gradients[sparsity][name+'.b_m'] = module.b_m.grad.clone() * module.bias_mask
                                            
                    # for name,param in self.model.named_parameters():
                    #     if 'projection_cls.last_layer' in name:
                    #         param.requires_grad = True
                                                
                    if self.cfg.SUB_MODEL.USE_SWITCHNET:
                        if self.cfg.COMP_LOSS.TYPE in ['independent','BCT']:
                            self.model.apply(lambda m: setattr(m, 'width_mult', self.cfg.SNET.WIDTH_MULT_LIST[0]))
                        else:
                            self.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                        feat, _, cls_score = self.model(images.cuda(), return_feature=True)
                    else:
                        feat, _, cls_score = self.model(images.cuda(), sparsity=0, mask=None, return_feature=True)
                        
                    loss = self.criterion['base'](cls_score, labels)
                                                                                                   
                    if self.cfg.SUB_MODEL.GRAD_PROJ:
                        self.optimizer.zero_grad()
                        self.grad_scaler.scale(loss).backward(retain_graph=False)

                        for name, module in self.model.named_modules():
                                if isinstance(module,(SubnetConv2d,SubnetLinear)):
                                    gradients['parent'][name+'.weight'] = module.weight.grad.clone()
                                    if module.bias is not None:
                                        gradients['parent'][name+'.bias'] = module.bias.grad.clone()
                                    if not self.cfg.SUB_MODEL.FREEZEW_M and self.cfg.SUB_MODEL.PROJ_W_M and not self.cfg.SUB_MODEL.USE_SWITCHNET:
                                        gradients['parent'][name+'.w_m'] = module.w_m.grad.clone()
                                        if module.bias is not None:
                                            gradients['parent'][name+'.b_m'] = module.b_m.grad.clone()
                                            
                    for name, module in self.model.named_modules():
                        proj_condition = isinstance(module,(SubnetConv2d,SubnetLinear))
                        if proj_condition:
                            for _name,param in module.named_parameters():
                                full_name = name + '.' + _name
                                if self.cfg.SUB_MODEL.FREEZEW_M:
                                    if 'w_m' in full_name or 'b_m' in full_name:
                                        continue
                                g_list = []
                                g_ret_list = []
                                g_list.append(gradients['parent'][full_name])
                                for idx, sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                                    if self.cfg.SUB_MODEL.USE_SWITCHNET:
                                        sparsity = self.cfg.SNET.WIDTH_MULT_LIST[idx]
                                    g_list.append(gradients[sparsity][full_name])

                                g_list_ori = [i.detach() for i in g_list]
                                if isinstance(module,(SubnetConv2d)) and len(param.shape)==4 and self.cfg.SUB_MODEL.PROJ_BY_KERNEL:
                                    g_shuffle = random.sample(g_list, len(g_list))
                                    for g_i in g_list:
                                        g_tmp = g_i.clone()
                                        for g_j in g_shuffle:
                                            g_i_flatten = g_i.view(g_i.shape[0], -1)
                                            g_j_flatten = g_j.view(g_i.shape[0], -1)
                                            vec_dot_product = torch.sum(g_i_flatten * g_j_flatten, dim=1)
                                            negative_indices = torch.where(vec_dot_product < 0)[0].tolist()
                                                # self.conflict_kernal = negative_indices
                                            if len(negative_indices) >0:
                                                tmp_i = g_i[negative_indices,:,:,:]
                                                tmp_j = g_j[negative_indices,:,:,:]
                                                tmp_j_flatten = tmp_j.view(len(negative_indices), -1)
                                                scale = vec_dot_product[negative_indices] / (torch.sum(tmp_j_flatten * tmp_j_flatten, dim=1)+1e-8)
                                                scale_expanded = scale.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(tmp_i)
                                                g_i_proj = tmp_i - scale_expanded * tmp_j
                                                g_tmp[negative_indices,:,:,:] = g_i_proj
                                        g_ret_list.append(g_tmp.detach())
                                else:
                                    for g_i in g_list:
                                        g_tmp = g_i.clone()
                                        g_shuffle = random.sample(g_list, len(g_list))
                                        for g_j in g_shuffle: 
                                            d = torch.dot(g_i.flatten(), g_j.flatten())  
                                            if d < 0:
                                                g_tmp_i = g_i
                                                g_tmp_j = g_j
                                                g_tmp = (g_tmp_i - d / (torch.dot(g_tmp_j.view(-1), g_tmp_j.view(-1))+1e-8) * g_tmp_j)
                                        g_ret_list.append(g_tmp.detach())

                                total_len = len(self.cfg.SUB_MODEL.SPARSITY)+1
                                cos_sim = []
                                for i in range(total_len):
                                    flat_tensor1 = g_ret_list[i].view(-1)
                                    flat_tensor2 = g_list_ori[i].view(-1)                                            
                                    if self.cfg.SUB_MODEL.COS_SIM.USE_DOT:
                                        dot_sim = torch.dot(flat_tensor1.view(-1), flat_tensor2.view(-1))
                                        cos_sim.append((dot_sim+1)/2)
                                    else:
                                        cos_sim_i = torch.nn.functional.cosine_similarity(flat_tensor1.unsqueeze(0), flat_tensor2.unsqueeze(0))
                                        cos_sim.append((cos_sim_i+1)/2)
                                g_tensor = torch.stack(g_ret_list,dim=0)
                                cos_sim = torch.tensor(cos_sim, dtype=torch.float32).to(self.device)
                                cos_sim = cos_sim ** self.cfg.SUB_MODEL.COS_SIM.EXP
                                cos_sim = cos_sim / (cos_sim.sum()+1e-8)
                                g_ratio = cos_sim
                                # print(cos_sim)
                                # g_ratio = cos_sim.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                                for _ in range(len(g_tensor.shape) - 1):
                                    g_ratio = g_ratio.unsqueeze(1)
                                g_ratio = g_ratio.expand_as(g_tensor)
                                g_ret = torch.sum(g_tensor * g_ratio, dim=0) * total_len
                                param.grad = g_ret 


                elif self.cfg.COMP_LOSS.TYPE in ['proj_with_cosine_sim_switchnet']:
                    compatible_loss_all = 0.0
                    count = 0
                    sub_loss = []
                    gradients = {}
                    if self.cfg.SUB_MODEL.GRAD_PROJ:
                        gradients['parent'] = {}
                    self.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                    feat, _, cls_score = self.model(images.cuda(), return_feature=True)
                    loss = self.criterion['base'](cls_score, labels)     
                    
                    if self.cfg.SUB_MODEL.GRAD_PROJ:
                        self.optimizer.zero_grad()
                        self.grad_scaler.scale(loss).backward(retain_graph=False)

                        for name, module in self.model.named_modules():
                                if isinstance(module,(SubnetConv2d,SubnetLinear,SlimmableConv2d,SlimmableLinear)):
                                    gradients['parent'][name+'.weight'] = module.weight.grad.clone()
                                    if module.bias is not None:
                                        gradients['parent'][name+'.bias'] = module.bias.grad.clone()   
                                elif 'projection_cls' in name:
                                    gradients['parent'][name+'.weight'] = module.weight.grad.clone()
                                    if module.bias is not None:
                                        gradients['parent'][name+'.bias'] = module.bias.grad.clone() 
                                                                                                                                                                                                         
                    for idx,sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                        if not self.cfg.SUB_MODEL.USE_SWITCHNET:
                            _model = sub_model_list[idx]
                        else:
                            sparsity = self.cfg.SNET.WIDTH_MULT_LIST[idx]
                            self.model.apply(lambda m: setattr(m, 'width_mult', sparsity))
                        if self.cfg.SUB_MODEL.GRAD_PROJ:
                            compatible_loss = 0.0
                            gradients[sparsity] = {}

                        feat_sub, _, cls_score_sub = self.model(images.cuda(), return_feature=True)
                        if self.cfg.SUB_MODEL.SUB_MODEL_LOSS_TYPE =='cls':
                            _sub_loss = self.criterion['base'](cls_score_sub, labels)
                            sub_loss.append(_sub_loss)
                            if self.cfg.SUB_MODEL.ADD_LOSS_SCALE:
                                loss_scale = (1 - float(sparsity)) if not self.cfg.SUB_MODEL.USE_SWITCHNET else (1-float(sparsity)**2)
                                _sub_loss *= loss_scale
                            compatible_loss += _sub_loss
                            compatible_loss_all += compatible_loss
                        elif self.cfg.SUB_MODEL.SUB_MODEL_LOSS_TYPE =='comp':
                            feat_sub_norm = F.normalize(feat_sub, dim=1, p=2.0)
                            # feat_norm = F.normalize(feat, dim=1, p=2.0)
                            _sub_loss = self.criterion["back_comp"](feat_sub_norm, labels, feat_norm.detach(), labels)
                            sub_loss.append(_sub_loss)
                            compatible_loss += _sub_loss
                            compatible_loss_all += compatible_loss                        
                                 
                        self.optimizer.zero_grad()
                        self.grad_scaler.scale(compatible_loss).backward(retain_graph=False)
                        torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=100)                            
                        if self.cfg.SUB_MODEL.GRAD_PROJ:
                            for name, module in _model.named_modules():
                                if isinstance(module,(SubnetConv2d,SubnetLinear,SlimmableConv2d,SlimmableLinear)):
                                    gradients[sparsity][name+'.weight'] = module.weight.grad.clone()
                                    if module.bias is not None:
                                        gradients[sparsity][name+'.bias'] = module.bias.grad.clone()
                                # elif 'projection_cls.last_layer' in name:
                                #     gradients[sparsity][name+'.weight'] =  module.weight.grad.clone()
                                #     if module.bias is not None:
                                #         gradients[sparsity][name+'.bias'] = module.bias.grad.clone() 
                                                                    
                    for name, module in self.model.named_modules():
                        proj_condition = isinstance(module,(SubnetConv2d,SubnetLinear,SlimmableConv2d,SlimmableLinear))
                        if proj_condition:
                            for _name,param in module.named_parameters():
                                full_name = name + '.' + _name
                                g_list = []
                                g_ret_list = []
                                g_list.append(gradients['parent'][full_name])
                                for idx, sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                                    if self.cfg.SUB_MODEL.USE_SWITCHNET:
                                        sparsity = self.cfg.SNET.WIDTH_MULT_LIST[idx]
                                    g_list.append(gradients[sparsity][full_name])
                                       
                                g_list_ori = [i.detach() for i in g_list]
                                if isinstance(module,(SubnetConv2d,SlimmableConv2d)) and len(param.shape)==4 and self.cfg.SUB_MODEL.PROJ_BY_KERNEL:
                                    g_shuffle = random.sample(g_list, len(g_list))
                                    for g_i in g_list:
                                        g_tmp = g_i.clone()
                                        for g_j in g_shuffle:
                                            g_i_flatten = g_i.view(g_i.shape[0], -1)
                                            g_j_flatten = g_j.view(g_i.shape[0], -1)
                                            vec_dot_product = torch.sum(g_i_flatten * g_j_flatten, dim=1)
                                            negative_indices = torch.where(vec_dot_product < 0)[0].tolist()
                                            if len(negative_indices) >0:
                                                tmp_i = g_i[negative_indices,:,:,:]
                                                tmp_j = g_j[negative_indices,:,:,:]
                                                tmp_j_flatten = tmp_j.view(len(negative_indices), -1)
                                                scale = vec_dot_product[negative_indices] / (torch.sum(tmp_j_flatten * tmp_j_flatten, dim=1)+1e-8)
                                                scale_expanded = scale.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(tmp_i)
                                                g_i_proj = tmp_i - scale_expanded * tmp_j
                                                g_tmp[negative_indices,:,:,:] = g_i_proj
                                        g_ret_list.append(g_tmp.detach())
                                else:
                                    for g_i in g_list:
                                        g_tmp = g_i.clone()
                                        g_shuffle = random.sample(g_list, len(g_list))
                                        for g_j in g_shuffle: 
                                            d = torch.dot(g_i.flatten(), g_j.flatten())  
                                            if d < 0:
                                                g_tmp_i = g_i
                                                g_tmp_j = g_j
                                                g_tmp = (g_tmp_i - d / (torch.dot(g_tmp_j.view(-1), g_tmp_j.view(-1))+1e-8) * g_tmp_j)
                                        g_ret_list.append(g_tmp.detach())


                                total_len = len(self.cfg.SUB_MODEL.SPARSITY)+1
                                cos_sim = []
                                for i in range(total_len):
                                    flat_tensor1 = g_ret_list[i].view(-1)
                                    flat_tensor2 = g_list_ori[i].view(-1)                                            
                                    if self.cfg.SUB_MODEL.COS_SIM.USE_DOT:
                                        dot_sim = torch.dot(flat_tensor1.view(-1), flat_tensor2.view(-1))
                                        cos_sim.append((dot_sim+1)/2)
                                    else:
                                        cos_sim_i = torch.nn.functional.cosine_similarity(flat_tensor1.unsqueeze(0), flat_tensor2.unsqueeze(0))
                                        cos_sim.append((cos_sim_i+1)/2)
                                g_tensor = torch.stack(g_ret_list,dim=0)
                                cos_sim = torch.tensor(cos_sim, dtype=torch.float32).to(self.device)
                                cos_sim = cos_sim ** self.cfg.SUB_MODEL.COS_SIM.EXP
                                cos_sim = cos_sim / (cos_sim.sum()+1e-8)
                                g_ratio = cos_sim
                                # print(cos_sim)
                                # g_ratio = cos_sim.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                                for _ in range(len(g_tensor.shape) - 1):
                                    g_ratio = g_ratio.unsqueeze(1)
                                g_ratio = g_ratio.expand_as(g_tensor)
                                g_ret = torch.sum(g_tensor * g_ratio, dim=0) * total_len
                                param.grad = g_ret
                                                         
                        elif 'projection_cls.last_layer' in name:
                            for _name,param in module.named_parameters():
                                full_name = name + '.' + _name
                                g_ret = gradients['parent'][full_name]
                                param.grad = g_ret                                   
                                                                                        
                elif self.cfg.COMP_LOSS.TYPE in ['SFSC']:
                    compatible_loss = 0.0
                    # assert self.cfg.NUM_GPUS == 1, "Can only one gpu."
                    tot_cls =  []
                    sub_loss = []
                    losses={}
                    TWC = TMC(self.cfg.MODEL.DATA.NUMBER_CLASSES)
                    grads, shapes, has_grads = [], [], []
                    if not self.cfg.SUB_MODEL.USE_SWITCHNET:
                        sub_model_list = build_sub_models(cfg=self.cfg,sparsity=self.cfg.SUB_MODEL.SPARSITY if not self.cfg.SUB_MODEL.RANDOM_SPARSITY else [torch.rand(1)],model=self.model)
                        for idx,sub_model in enumerate(sub_model_list):
                            sparsity = self.cfg.SUB_MODEL.SPARSITY[idx]
                            feat_sub, _, cls_score_sub = sub_model(images.cuda(), sparsity=sparsity, mask=None,return_feature=True)
                            tot_cls.append(cls_score_sub)
                    else: # swtichnet
                        
                        # for name, param in self.model.named_parameters():
                            # if 'projection_cls' in name:# used to be projection_cls
                                # param.requires_grad = True
                                # print(param)
                        feat, _, cls_score = self.model(images.cuda(), return_feature=True)
                        loss = self.criterion['base'](cls_score, labels)
                        losses['loss_ori']=loss
                        self.optimizer.zero_grad()
                        self.grad_scaler.scale(loss).backward(retain_graph=True)
                        grad, shape, has_grad = self.pcg._retrieve_grad()
                        grads.append(self.pcg._flatten_grad(grad, shape))
                        has_grads.append(self.pcg._flatten_grad(has_grad, shape))
                        shapes.append(shape)
                        
                        # for name, param in self.model.named_parameters():
                        #     if 'projection_cls' in name:# used to be projection_cls
                        #         param.requires_grad = False
                                # print(param.grad)
                                
                        for i in range(len(self.cfg.SNET.WIDTH_MULT_LIST)-1):
                            mult = self.cfg.SNET.WIDTH_MULT_LIST[i]
                            self.model.apply(lambda m: setattr(m, 'width_mult', mult))
                            feat_sub, _, cls_score_sub = self.model(images.cuda(), return_feature=True)
                            if not self.cfg.SNET.USE_SFSC_LOSS:
                                _sub_model_cls_loss = self.criterion["base"](cls_score_sub, labels)
                                sub_loss.append(_sub_model_cls_loss)
                                losses['sub'+str(mult)] = _sub_model_cls_loss
                                self.optimizer.zero_grad()
                                self.grad_scaler.scale(_sub_model_cls_loss).backward(retain_graph=True)
                                grad, shape, has_grad = self.pcg._retrieve_grad()
                                grads.append(self.pcg._flatten_grad(grad, shape))
                                has_grads.append(self.pcg._flatten_grad(has_grad, shape))
                                shapes.append(shape)
                                
                            tot_cls.append(cls_score_sub)
                    if self.cfg.SNET.USE_SFSC_LOSS:
                        _lambda = self.cfg.SNET.LAMBDA
                        tmp = TWC(tot_cls, labels, _lambda)
                        compatible_loss = torch.mean(torch.stack(list(tmp.values()), dim=0))                
                        for i in tmp:
                            losses[i]=tmp[i]
                            sub_loss.append(tmp[i])                        
                    # for name, param in self.model.named_parameters():
                    #     if 'projection_cls' in name:# used to be projection_cls
                    #         param.requires_grad = True                            
                    self.pcg.zero_grad()
                    self.pcg.pc_backward(losses, grads, shapes, has_grads)
                    self.pcg.step()
                    # for name, param in self.model.named_parameters():
                    #     if 'projection_cls' in name:# used to be projection_cls
                    #         param.requires_grad = True           
                               
                elif self.cfg.COMP_LOSS.TYPE in ['BCT_S']:
                    compatible_loss = 0.0
                    sub_loss = []
                    sub_model_cls_loss = 0.0
                    if self.cfg.SUB_MODEL.USE_SWITCHNET:
                        for idx,sparsity in enumerate(self.cfg.SNET.WIDTH_MULT_LIST):
                            if sparsity != 1.0:
                                self.model.apply(lambda m: setattr(m, 'width_mult', sparsity))
                                feat_sub, _, cls_score_sub = self.model(images.cuda(),return_feature=True)
                                _sub_model_cls_loss = self.criterion["base"](cls_score_sub, labels)
                                sub_loss.append(_sub_model_cls_loss)
                                sub_model_cls_loss = sub_model_cls_loss + _sub_model_cls_loss
                    else:                                      
                        for idx,sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                            feat_sub, _, cls_score_sub = self.model(images.cuda(), sparsity=sparsity, mask=None,return_feature=True)
                            _sub_model_cls_loss = self.criterion["base"](cls_score_sub, labels)
                            sub_loss.append(_sub_model_cls_loss)
                            sub_model_cls_loss = sub_model_cls_loss + _sub_model_cls_loss  
                
                elif self.cfg.COMP_LOSS.TYPE in ['BCT']:
                    compatible_loss = 0.0
                    sub_loss = []
                    sub_model_cls_loss = 0.0      
                    if self.cfg.SUB_MODEL.USE_SWITCHNET:
                        cls_parent = self.parent_model.module.projection_cls if self.cfg.NUM_GPUS > 1 else self.parent_model.projection_cls
                        for idx,sparsity in enumerate(self.cfg.SNET.WIDTH_MULT_LIST):
                            self.model.apply(lambda m: setattr(m, 'width_mult', sparsity))
                            feat_sub, _, _ = self.model(images.cuda(),return_feature=True)
                            cls_score_bct = cls_parent(feat_sub)
                            _sub_model_cls_loss = self.criterion["base"](cls_score_bct, labels)
                            sub_loss.append(_sub_model_cls_loss)
                            sub_model_cls_loss = sub_model_cls_loss + _sub_model_cls_loss
                    else:                                      
                        for idx,sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                            feat_sub, _, _ = self.model(images.cuda(), sparsity=sparsity, mask=None,return_feature=True)
                            cls_score_bct = cls_parent(feat_sub)
                            _sub_model_cls_loss = self.criterion["base"](cls_score_bct, labels)
                            sub_loss.append(_sub_model_cls_loss)
                            sub_model_cls_loss = sub_model_cls_loss + _sub_model_cls_loss
                                                                  
                else:
                    raise NotImplementedError
                
            compatible_loss = self.cfg.SUB_MODEL.COM_LOSS_SCALE * compatible_loss
            loss_back_comp_value = tensor_to_float(compatible_loss)
            # losses_supcontrast_comp.update(loss_back_comp_value, len(labels))
            if sub_loss:
                for idx,i in enumerate(sub_loss):
                    sub_loss_value = tensor_to_float(i)
                    submodel_loss[idx].update(sub_loss_value, len(labels)) 

                        
                        

            if loss != 0:
                losses_cls.update(loss.item(), images.size(0))
            else:
                losses_cls.update(0.0, images.size(0))
            if self.cfg.SUB_MODEL.BCT_S:
                sub_model_cls_loss_value = tensor_to_float(sub_model_cls_loss)
                losses_sub_model_cls.update(sub_model_cls_loss_value, len(labels))

            if self.device == 0:
                self.writer.add_scalar("Cls loss", losses_cls.avg, total_steps)
                # self.writer.add_scalar("Submodel loss", losses_supcontrast_comp.avg, total_steps)  
                if self.cfg.SUB_MODEL.BCT_S:
                    self.writer.add_scalar("Sub model cls loss", losses_sub_model_cls.avg, total_steps)
                if self.cfg.COMP_LOSS.TYPE not in ['independent']:
                    for idx,sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                        self.writer.add_scalar(f"Sub model {sparsity} cls loss", submodel_loss[idx].avg, total_steps)  

                
            if self.cfg.LOSS.ONLY_COMPATIBLE:
                loss = compatible_loss
            elif self.cfg.COMP_LOSS.TYPE in ['BCT_S','BCT']:
                loss = loss + sub_model_cls_loss
                # print(loss)
            elif self.cfg.COMP_LOSS.TYPE == 'independent':
                loss = loss
            elif self.cfg.COMP_LOSS.TYPE in ['proj_with_cosine_sim']:
                loss = loss + compatible_loss_all
            else:
                loss = loss + compatible_loss

            losses_all.update(loss.item(), images.size(0))
            
            if self.device == 0:
                self.writer.add_scalar("Total loss", losses_all.avg, total_steps)
            # compute gradient and do SGD step
            if self.cfg.COMP_LOSS.TYPE not in ['SFSC']:   
                if not self.cfg.SUB_MODEL.GRAD_PROJ:
                    self.optimizer.zero_grad()
                    self.grad_scaler.scale(loss).backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)   #梯度裁剪 防止nan
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            if not self.cfg.SOLVER.OPTIMIZER == 'SGD':
                self.scheduler.step()
       
            if batch_idx % self.cfg.SOLVER.LOG_EVERY_N == 0:
                if self.cfg.SOLVER.BACKBONE_MULTIPLIER != 1.:
                    progress.display(batch_idx, suffix=f"\tgroup0 lr:{self.optimizer.param_groups[0]['lr']:.9f}" + f"\tgroup1 lr:{self.optimizer.param_groups[2]['lr']:.9f}")
                else:
                    progress.display(batch_idx, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.9f}")
            if self.device == 0:
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], total_steps)

                
