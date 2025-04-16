#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss
from .utils import cosine_dist, euclidean_dist, hard_example_mining, weighted_example_mining
import torch.distributed as dist

from ..utils import logging
logger = logging.get_logger("FCLearning")


class SigmoidLoss(nn.Module):
    def __init__(self, cfg=None):
        super(SigmoidLoss, self).__init__()

    def is_single(self):
        return True

    def is_local(self):
        return False

    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        # (batch_size, num_classes)
        return target

    def loss(
        self, logits, targets, per_cls_weights,
        multihot_targets: Optional[bool] = False
    ):
        # targets: 1d-tensor of integer
        # Only support single label at this moment
        # if len(targets.shape) != 2:
        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        # logger.info(f"loss shape: {loss.shape}")
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        ).unsqueeze(0)
        # logger.info(f"weight shape: {weight.shape}")
        loss = torch.mul(loss.to(torch.float32), weight.to(torch.float32))
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, targets, per_cls_weights, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  per_cls_weights, multihot_targets)
        return loss


class SoftmaxLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(SoftmaxLoss, self).__init__()

    def loss(self, logits, targets, per_cls_weights, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        return torch.sum(loss) / targets.shape[0]


class CustomContrastiveLoss(ContrastiveLoss):
    """Training Vision Transformers for Image Retrieval equation (1)
    """
    def __init__(self, cfg = None, distance=CosineSimilarity(),**kwargs):
        super(CustomContrastiveLoss, self).__init__(
            pos_margin=cfg.SOLVER.LOSS_POS_MARGIN,
            neg_margin=cfg.SOLVER.LOSS_NEG_MARGIN,
            distance=distance,
            **kwargs)

class Tripletloss(nn.Module):
    def __init__(self, cfg=None):
        super(Tripletloss, self).__init__()
        self.cfg = cfg

    def triplet_loss(self,embedding, targets,norm_feat = True):
        margin = self.cfg.SOLVER.LOSS_MARGIN 
        hard_mining = self.cfg.SOLVER.LOSS_HARD_MINING
        if norm_feat:
            dist_mat = cosine_dist(embedding, embedding)
        else:
            dist_mat = euclidean_dist(embedding, embedding)

        # For distributed training, gather all features from different process.
        # if comm.get_world_size() > 1:
        #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
        #     all_targets = concat_all_gather(targets)
        # else:
        #     all_embedding = embedding
        #     all_targets = targets

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        if hard_mining:
            dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
        else:
            dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)
        #y代表真实标签
        y = dist_an.new().resize_as_(dist_an).fill_(1)


        if margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            # fmt: off
            if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # fmt: on

        return loss
    def forward(self,embedding, targets,  norm_feat = True):

        loss = self.triplet_loss(embedding, targets,  norm_feat)

        return loss
    
class HotRefreshLoss(nn.Module):
    def __init__(self, cfg=None, temp=0.01, margin=0.8, topk_neg=30,
                 loss_type='contra', loss_weight=1.0, gather_all=False):
        super(HotRefreshLoss, self).__init__()
        self.temperature = temp
        self.loss_weight = loss_weight
        self.topk_neg = topk_neg

        #   loss_type options:
        #   - contra
        #   - triplet
        #   - l2
        #   - contra_ract (paper: "")
        #   - triplet_ract
        if loss_type in ['contra', 'contra_ract']:
            self.criterion = nn.CrossEntropyLoss().cuda()
        elif loss_type in ['triplet', 'triplet_ract']:
            assert topk_neg > 0, \
                "Please select top-k negatives for triplet loss"
            # not use nn.TripletMarginLoss()
            self.criterion = nn.MarginRankingLoss(margin=margin).cuda()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss().cuda()
        else:
            raise NotImplementedError("Unknown loss type: {}".format(loss_type))
        self.loss_type = loss_type
        self.gather_all = gather_all

    def forward(self, feat, feat_old, targets):
        # features l2-norm
        feat = F.normalize(feat, dim=1, p=2)
        
        feat_old = F.normalize(feat_old, dim=1, p=2).detach()
        batch_size = feat.size(0)
        # gather tensors from all GPUs
        if self.gather_all:
            feat_large = gather_tensor(feat)
            feat_old_large = gather_tensor(feat_old)
            targets_large = gather_tensor(targets)
            batch_size_large = feat_large.size(0)
            current_gpu = dist.get_rank()
            masks = targets_large.expand(batch_size_large, batch_size_large) \
                .eq(targets_large.expand(batch_size_large, batch_size_large).t())
            masks = masks[current_gpu * batch_size: (current_gpu + 1) * batch_size, :]  # size: (B, B*n_gpus)
        else:
            feat_large, feat_old_large = None, None
            masks = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())

        # compute loss
        loss_comp = calculate_loss(feat, feat_old, feat_large, feat_old_large, masks, self.loss_type, self.temperature,
                                   self.criterion, self.loss_weight, self.topk_neg)
        return loss_comp



class SupContrastiveLoss(nn.Module):
    
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, cfg, temperature=0.07, base_temperature=0.07):
        super(SupContrastiveLoss, self).__init__()
        self.cfg = cfg
        self.temperature = temperature
        self.base_temperature = base_temperature

    def _calculate_loss_within_batch(self, features, labels):
        device = features.device
        features = features.unsqueeze(1)
        batch_size = features.shape[0]
        #print('labels', labels)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature# all view
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss
    
    def _calculate_loss_with_mem(self, features, labels, img_ids, ref_features, ref_labels, ref_img_ids):
        # pro
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        anchor_feature = features

        mem_size = ref_features.shape[0]
        ref_labels = ref_labels.contiguous().view(-1, 1)
        if ref_labels.shape[0] != mem_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, ref_labels.T).float().to(device)
        contrast_feature = ref_features
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数
        
        # mask-out self-contrast cases
        img_ids = img_ids.contiguous().view(-1, 1)
        ref_img_ids = ref_img_ids.contiguous().view(-1, 1)
        logits_mask = 1 - torch.eq(img_ids, ref_img_ids.T).float().to(device)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss

    def _calculate_loss_with_joint(self, features, labels, img_ids, ref_features, ref_labels, ref_img_ids, use_ref_pos):
        # pro
        raise NotImplementedError
        device = features.device
        features = features.unsqueeze(1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        ref_features = ref_features.unsqueeze(1)
        mem_size = ref_features.shape[0]
        ref_labels = ref_labels.contiguous().view(-1, 1)
        if ref_labels.shape[0] != mem_size:
            raise ValueError('Num of labels does not match num of features')
        ref_mask = torch.eq(labels, ref_labels.T).float().to(device)
        ref_contrast_feature = torch.cat(torch.unbind(ref_features, dim=1), dim=0)
        contrast_feature = torch.cat([anchor_feature, ref_contrast_feature], dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        if use_ref_pos:
            img_ids = img_ids.contiguous().view(-1, 1)
            ref_img_ids = ref_img_ids.contiguous().view(-1, 1)
            ref_logits_mask = 1 - torch.eq(img_ids, ref_img_ids.T).float().to(device)
            #ref_logits_mask = torch.ones_like(ref_mask).to(device)
            print('joint remove same img id')
        else:
            ref_logits_mask = 1 - ref_mask# 只考虑ref_features中的负样本
        mask = torch.cat([mask, ref_mask], dim=1)
        logits_mask = torch.cat([logits_mask, ref_logits_mask], dim=1)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss


    
    def forward(self, features, labels, img_ids=None, ref_features=None, ref_labels=None, ref_img_ids=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, f_dim].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if ref_features is None:
            loss = self._calculate_loss_within_batch(features, labels)
        else:
            loss = self._calculate_loss_with_mem(features, labels, img_ids, 
                                                 ref_features, ref_labels, ref_img_ids)
        
        return loss
    
class Prototype_loss(nn.Module):
    def __init__(self, cfg=None, loss_type='contra'):
        super(Prototype_loss,self).__init__()
        self.cfg = cfg
        self.loss_type = loss_type
        if self.loss_type in ['contra', 'contra_ract']:
            self.criterion = nn.CrossEntropyLoss().cuda()
        elif loss_type in ['triplet', 'triplet_ract']:
            '''
            assert topk_neg > 0, \
                "Please select top-k negatives for triplet loss"
            # not use nn.TripletMarginLoss()
            '''
            self.criterion = nn.MarginRankingLoss(margin=cfg.SOLVER.LOSS_POS_MARGIN).cuda()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss().cuda()
        else:
            raise NotImplementedError("Unknown loss type: {}".format(loss_type))
    def forward(self,embedding,old_prototype,targets):

        loss_comp = calculate_loss_prototype(embedding,old_prototype,targets,self.criterion)

        return loss_comp

class CompatibleSupContrastiveLoss(nn.Module):
    
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, cfg, temperature=0.07, base_temperature=0.07):
        super(CompatibleSupContrastiveLoss, self).__init__()
        self.cfg = cfg
        self.temperature = temperature
        self.base_temperature = base_temperature

    def _cal_loss_wo_new(self, features, labels, old_features, old_labels):
        # contrastive features 只有 old_features
        device = features.device
        batch_size = features.shape[0]
        mem_size = old_features.shape[0]# 考虑old_features使用XBM
        
        labels = labels.contiguous().view(-1, 1)
        old_labels = old_labels.contiguous().view(-1, 1)
        
        if (labels.shape[0] != batch_size) or (old_labels.shape[0] != mem_size):
            raise ValueError('Num of (old) labels does not match num of (old) features')
        
        mask = torch.eq(labels, old_labels.T).float().to(device)
        
        contrast_feature = old_features
        anchor_feature = features# all view
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        #logits_mask = torch.ones_like(mask).to(device)# 自己和自己比也应该保留
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss
    
    def _cal_loss_wi_new(self, features, labels, old_features, old_labels, img_ids=None, old_img_ids=None):
        # contrastive features 包括 old_features, new_features, xbm_old_features
        device = features.device
        batch_size = features.shape[0]
        mem_size = old_features.shape[0]# 考虑old_features使用XBM
        
        labels = labels.contiguous().view(-1, 1)
        old_labels = old_labels.contiguous().view(-1, 1)
        
        if (labels.shape[0] != batch_size) or (old_labels.shape[0] != mem_size):
            raise ValueError('Num of (old) labels does not match num of (old) features')
        
        old_mask = torch.eq(labels, old_labels.T).float().to(device)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat([old_features, features], dim=0)
        anchor_feature = features# all view
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数

        # mask-out self-contrast cases
        if img_ids is None and old_img_ids is None:
            old_logits_mask = torch.scatter(
                torch.ones_like(old_mask), 1, torch.arange(mem_size).view(-1, 1).to(device), 0)
        else:
            img_ids = img_ids.contiguous().view(-1, 1)
            old_img_ids = old_img_ids.contiguous().view(-1, 1)
            old_logits_mask = 1 - torch.eq(img_ids, old_img_ids.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = torch.cat([old_mask, mask], dim=1)
        logits_mask = torch.cat([old_logits_mask, logits_mask], dim=1)
        mask = mask * logits_mask
        #print("mask.sum(1): ", mask.sum(1))

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss
    
    def _cal_loss_wi_newxbm(self, features, labels, old_features, old_labels, newxbm_features, newxbm_labels, 
                            img_ids, old_img_ids, newxbm_img_ids):
        # contrastive features 包括 old_features, new_features, xbm_old_features
        device = features.device
        batch_size = features.shape[0]
        old_mem_size = old_features.shape[0]# 考虑old_features使用XBM
        new_mem_size = newxbm_features.shape[0]# new_features使用XBM

        labels = labels.contiguous().view(-1, 1)
        old_labels = old_labels.contiguous().view(-1, 1)
        newxbm_labels = newxbm_labels.contiguous().view(-1, 1)
        
        if (labels.shape[0] != batch_size) or (newxbm_labels.shape[0] != new_mem_size) or (old_labels.shape[0] != old_mem_size):
            raise ValueError('Num of (old) labels does not match num of (old) features')
        
        old_mask = torch.eq(labels, old_labels.T).float().to(device)
        newmem_mask = torch.eq(labels, newxbm_labels.T).float().to(device)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat([old_features, newxbm_features, features], dim=0)
        anchor_feature = features# all view
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数

        # mask-out self-contrast cases
        img_ids = img_ids.contiguous().view(-1, 1)
        old_img_ids = old_img_ids.contiguous().view(-1, 1)
        newxbm_img_ids = newxbm_img_ids.contiguous().view(-1, 1)
        old_logits_mask = 1 - torch.eq(img_ids, old_img_ids.T).float().to(device)
        newxbm_logits_mask = 1 - torch.eq(img_ids, newxbm_img_ids.T).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = torch.cat([old_mask, newmem_mask, mask], dim=1)
        logits_mask = torch.cat([old_logits_mask, newxbm_logits_mask, logits_mask], dim=1)
        mask = mask * logits_mask
        #print("mask.sum(1): ", mask.sum(1))

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        return loss

    def forward(self, features, labels, old_features, old_labels, newxbm_features=None, newxbm_labels=None, 
                img_ids=None, old_img_ids=None, newxbm_img_ids=None, withnew=False, withnewxbm=False):
        if not withnew:
            return self._cal_loss_wo_new(features, labels, old_features, old_labels, img_ids, old_img_ids)
        elif withnew and (not withnewxbm):# 兼容没有old-xbm的情况
            return self._cal_loss_wi_new(features, labels, old_features, old_labels, img_ids, old_img_ids)
        elif withnewxbm:
            return self._cal_loss_wi_newxbm( features, labels, old_features, old_labels, newxbm_features, newxbm_labels, 
                            img_ids, old_img_ids, newxbm_img_ids)

    
LOSS = {
    "softmax": SoftmaxLoss,
    "contrastive": CustomContrastiveLoss,
    "triplet": Tripletloss,
    "supcontrastive": SupContrastiveLoss,
}

COMPATIBLE_LOSS = {
    "softmax": SoftmaxLoss,
    "supcontrastive": CompatibleSupContrastiveLoss,
    "contrastive": CustomContrastiveLoss,
    "triplet": Tripletloss,
    "hotrefreshloss": HotRefreshLoss, #BackwardCompatibleLoss
}


def build_loss(cfg):
    loss_func = {}
    for loss_name in cfg.SOLVER.LOSS:
        assert loss_name in LOSS, \
            f'loss name {loss_name} is not supported'
        loss_fn = LOSS[loss_name]
        if not loss_fn:
            pass
        else:
            loss_func[loss_name] = loss_fn(cfg)
    return loss_func

def build_compatible_loss(cfg):
    loss_func = {}
    for loss_name in cfg.COMPATIBLE.LOSS:
        assert loss_name in COMPATIBLE_LOSS, \
            f'loss name {loss_name} is not supported'
        loss_fn = COMPATIBLE_LOSS[loss_name]
        if not loss_fn:
            pass
        else:
            loss_func[loss_name] = loss_fn(cfg)
    return loss_func

def calculate_loss_prototype(feat_new, old_prototype,targets,criterion,new_prototype = None,loss_type = 'contra_old' ,temp = 0.5,
                   loss_weight=1.0):
    '''
    Args::
        feat_new: 新模型提取的特征
        old_prototype: 旧模型提取的原型,大小BXC,C代表类别数
        targets:标签数据,大小Bx1
        criterion:交叉熵损失
        loss_type:损失函数类型
    Returns:
        loss:
        
    '''
    B, D = feat_new.shape
    C, D_old = old_prototype.shape
    
    if D_old < D:
        old_prototype.expand(C,D)
    else:
        feat_new.expand(B,D_old)
    
    #labels_idx = torch.arange(B) + torch.distributed.get_rank() * B
    
    if loss_type == 'contra_old':
        logits_n2o_all = torch.mm(feat_new , old_prototype.permute(1,0))  # B*C
        #logits_all /= temp
        loss = criterion(logits_n2o_all, targets) 
        

    elif loss_type == 'contra_new_old':
        array= np.random.randint(0,2,C)
        idx = torch.tensor(array)
        idx = idx.view(C,1).expand(C,D)
        idx_new = idx^1
        old_prototype = old_prototype*idx
        new_prototype = new_prototype*idx_new 
        property = torch.add(old_prototype,new_prototype)
        loss = criterion(property, targets) 

    else:
        loss = 0.

    return loss


def calculate_loss(feat_new, feat_old, feat_new_large, feat_old_large,
                   masks, loss_type, temp, criterion,
                   loss_weight=1.0, topk_neg=-1):
    B, D = feat_new.shape
    #labels_idx = torch.arange(B) + torch.distributed.get_rank() * B
    if feat_new_large is None:
        feat_new_large = feat_new
        feat_old_large = feat_old

    if loss_type == 'contra':
        logits_n2o_pos = torch.bmm(feat_new.view(B, 1, D), feat_old.view(B, D, 1))  # B*1
        logits_n2o_pos = torch.squeeze(logits_n2o_pos, 1)
        logits_n2o_neg = torch.mm(feat_new, feat_old_large.permute(1, 0))  # B*B
        logits_n2o_neg = logits_n2o_neg - masks * 1e9
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o_neg, topk_neg, dim=1)[0]
        logits_all = torch.cat((logits_n2o_pos, logits_n2o_neg), 1)  # B*(1+k)
        logits_all /= 1

        labels_idx = torch.zeros(B).long().cuda()
        loss = criterion(logits_all, labels_idx) * loss_weight

    elif loss_type == 'contra_ract':
        logits_n2o_pos = torch.bmm(feat_new.view(B, 1, D), feat_old.view(B, D, 1))  # B*1
        logits_n2o_pos = torch.squeeze(logits_n2o_pos, 1)
        logits_n2o_neg = torch.mm(feat_new, feat_old_large.permute(1, 0))  # B*B
        logits_n2o_neg = logits_n2o_neg - masks * 1e9
        logits_n2n_neg = torch.mm(feat_new, feat_new_large.permute(1, 0))  # B*B
        logits_n2n_neg = logits_n2n_neg - masks * 1e9
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o_neg, topk_neg, dim=1)[0]
            logits_n2n_neg = torch.topk(logits_n2n_neg, topk_neg, dim=1)[0]
        logits_all = torch.cat((logits_n2o_pos, logits_n2o_neg, logits_n2n_neg), 1)  # B*(1+2B)
        logits_all /= temp

        labels_idx = torch.zeros(B).long().cuda()
        loss = criterion(logits_all, labels_idx) * loss_weight

    elif loss_type == 'triplet':
        logits_n2o = euclidean_dist(feat_new, feat_old_large)
        logits_n2o_pos = torch.gather(logits_n2o, 1, labels_idx.view(-1, 1).cuda())

        # find the hardest negative
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o + masks * 1e9, topk_neg, dim=1, largest=False)[0]

        logits_n2o_pos = logits_n2o_pos.expand_as(logits_n2o_neg).contiguous().view(-1)
        logits_n2o_neg = logits_n2o_neg.view(-1)
        hard_labels_idx = torch.ones_like(logits_n2o_pos)
        loss = criterion(logits_n2o_neg, logits_n2o_pos, hard_labels_idx) * loss_weight

    elif loss_type == 'triplet_ract':
        logits_n2o = euclidean_dist(feat_new, feat_old_large)
        logits_n2o_pos = torch.gather(logits_n2o, 1, labels_idx.view(-1, 1).cuda())

        logits_n2n = euclidean_dist(feat_new, feat_new_large)
        # find the hardest negative
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o + masks * 1e9, topk_neg, dim=1, largest=False)[0]
            logits_n2n_neg = torch.topk(logits_n2n + masks * 1e9, topk_neg, dim=1, largest=False)[0]

        logits_n2o_pos = logits_n2o_pos.expand_as(logits_n2o_neg).contiguous().view(-1)
        logits_n2o_neg = logits_n2o_neg.view(-1)
        logits_n2n_neg = logits_n2n_neg.view(-1)
        hard_labels_idx = torch.ones_like(logits_n2o_pos)
        loss = criterion(logits_n2o_neg, logits_n2o_pos, hard_labels_idx)
        loss += criterion(logits_n2n_neg, logits_n2o_pos, hard_labels_idx)
        loss *= loss_weight

    elif loss_type == 'l2':
        loss = criterion(feat_new, feat_old) * loss_weight

    else:
        loss = 0.

    return loss

def gather_tensor(raw_tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensor_large = [torch.zeros_like(raw_tensor)
                    for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_large, raw_tensor.contiguous())
    tensor_large = torch.cat(tensor_large, dim=0)
    return tensor_large

class triplet_loss_n2o(nn.Module):
    
    def __init__(self, cfg):
        super(triplet_loss_n2o, self).__init__()
        self.cfg = cfg
    
    def forward(self,embedding_new, embedding_old, targets, margin, norm_feat, hard_mining):
                    
        """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
        Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
        Loss for Person Re-Identification'."""

        if norm_feat:
            dist_mat = cosine_dist(embedding_new, embedding_old)
        else:
            dist_mat = euclidean_dist(embedding_new, embedding_old)

        # For distributed training, gather all features from different process.
        # if comm.get_world_size() > 1:
        #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
        #     all_targets = concat_all_gather(targets)
        # else:
        #     all_embedding = embedding
        #     all_targets = targets

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        if hard_mining:
            dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
        else:
            dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            # fmt: off
            if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # fmt: on

        return loss