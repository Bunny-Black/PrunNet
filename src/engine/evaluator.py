#!/usr/bin/env python3
import time
from typing import Union, Tuple, List, Optional, Callable
import copy

import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

from PIL import ImageFile
from ..utils.meters import AverageMeter

from collections import defaultdict
from typing import List, Union
from tqdm import tqdm

from sklearn.metrics import average_precision_score
#from .eval.ranking import cmc, mean_ap_fct, mean_ap
from .eval import multilabel
from .eval import singlelabel
from ..utils import logging
logger = logging.get_logger("FCLearning")

class ResultWriter():
    def __init__(self,result_path):
        self.result_path = result_path
        with open(result_path,'w') as f:
            f.write('epoch mAP\n')

    def write(self,epoch,mAP):
        with open(self.result_path,'a') as f:
            f.write(f"{epoch} {mAP}\n")

class Evaluator():
    """
    An evaluator with below logics:

    1. find which eval module to use.
    2. store the eval results, pretty print it in log file as well.
    """

    def __init__(
        self,
    ) -> None:
        self.results = defaultdict(dict)
        self.iteration = -1
        self.threshold_end = 0.5

    def update_iteration(self, iteration: int) -> None:
        """update iteration info"""
        self.iteration = iteration

    def update_result(self, metric: str, value: Union[float, dict]) -> None:
        if self.iteration > -1:
            key_name = "epoch_" + str(self.iteration)
        else:
            key_name = "final"
        if isinstance(value, float):
            self.results[key_name].update({metric: value})
        else:
            if metric in self.results[key_name]:
                self.results[key_name][metric].update(value)
            else:
                self.results[key_name].update({metric: value})

    def classify(self, probs, targets, test_data, multilabel=False):
        """
        Evaluate classification result.
        Args:
            probs: np.ndarray for num_data x num_class, predicted probabilities
            targets: np.ndarray for multilabel, list of integers for single label
            test_labels:  map test image ids to a list of class labels
        """
        if not targets:
            raise ValueError(
                "When evaluating classification, need at least give targets")

        if multilabel:
            self._eval_multilabel(probs, targets, test_data)
        else:
            self._eval_singlelabel(probs, targets, test_data)

    def _eval_singlelabel(
        self,
        scores: np.ndarray,
        targets: List[int],
        eval_type: str
    ) -> None:
        """
        if number of labels > 2:
            top1 and topk (5 by default) accuracy
        if number of labels == 2:
            top1 and rocauc
        """
        acc_dict = singlelabel.compute_acc_auc(scores, targets)

        log_results = {
            k: np.around(v * 100, decimals=2) for k, v in acc_dict.items()
        }
        save_results = acc_dict

        self.log_and_update(log_results, save_results, eval_type)

    def _eval_multilabel(
        self,
        scores: np.ndarray,
        targets: np.ndarray,
        eval_type: str
    ) -> None:
        num_labels = scores.shape[-1]
        targets = multilabel.multihot(targets, num_labels)

        log_results = {}
        ap, ar, mAP, mAR = multilabel.compute_map(scores, targets)
        f1_dict = multilabel.get_best_f1_scores(
            targets, scores, self.threshold_end)

        log_results["mAP"] = np.around(mAP * 100, decimals=2)
        log_results["mAR"] = np.around(mAR * 100, decimals=2)
        log_results.update({
            k: np.around(v * 100, decimals=2) for k, v in f1_dict.items()})
        save_results = {
            "ap": ap, "ar": ar, "mAP": mAP, "mAR": mAR, "f1": f1_dict
        }
        self.log_and_update(log_results, save_results, eval_type)

    def log_and_update(self, log_results, save_results, eval_type):
        log_str = ""
        for k, result in log_results.items():
            if not isinstance(result, np.ndarray):
                log_str += f"{k}: {result:.2f}\t"
            else:
                log_str += f"{k}: {list(result)}\t"
        logger.info(f"Classification results with {eval_type}: {log_str}")
        # save everything
        self.update_result("classification", {eval_type: save_results})


class Evaluator_FCT(Evaluator):
    def __init__(self, cfg, query_model, gallery_model, device, old_model=None):
        super(Evaluator_FCT, self).__init__()
        #self.query_model = query_model
        #self.gallery_model = gallery_model
        self.old_model = old_model
        self.device = device
        self.cfg = cfg
        self.result_writer = ResultWriter(os.path.join(cfg.OUTPUT_DIR, 'eval_result.txt'))
        self.omst_best_mAP = 0.0# old model self-test
        self.nmst_best_mAP = 0.0# new model self-test
        self.ct_best_mAP = 0.0# new-old model cross-test
        self.omst_best_recall_1 = 0.0
        self.nmst_best_recall_1 = 0.0
        self.ct_best_recall_1 = 0.0        
    def evaluate(self, _model, _old_model, query_loader, gallery_loader, log_writer, epoch, mode="old model self-test"):
        if mode in ["old model self-test", "new model self-test"]:
            query_model = _model
            gallery_model = _model
            logger.info("=> same-model test")
        elif mode == "cross-test":
            query_model = _model
            gallery_model = _old_model
            logger.info("=> cross-model test")
        else:
            raise NotImplementedError
        distance_map = {"l2": l2_distance_matrix, "cosine": cosine_distance_matrix}
        
        feat_dim = _model.module.feat_dim if self.cfg.NUM_GPUS > 1 else _model.feat_dim# 获取backbone feature dim
        emb_dim = feat_dim if self.cfg.NEW_MODEL.PROJECTION_LAYERS < 0 else self.cfg.NEW_MODEL.PROJECTION_DIM# 判断是否有投影层
        cross_test_flag = False

        if _old_model is not None:
            old_feat_dim = _old_model.module.feat_dim if self.cfg.NUM_GPUS > 1 else _old_model.feat_dim# 获取old backbone feature dim
            old_emb_dim = old_feat_dim if self.cfg.MODEL.PROJECTION_LAYERS < 0 else self.cfg.MODEL.PROJECTION_DIM# 判断是否有投影层
            cross_test_flag = True

        distance_metric = distance_map.get(self.cfg.EVAL.DISTANCE_NAME.lower())
        logger.info('Generating Feature Matrix...')
        query_features, query_labels = generate_feature_matrix(self.cfg, query_model, query_loader, self.device, desc="query (new model)")
        if gallery_loader is None:
            if mode in ["old model self-test", "new model self-test"]:
                gallery_features = copy.deepcopy(query_features)
                gallery_labels = None
            elif mode == "cross-test":
                gallery_features, _ = generate_feature_matrix(self.cfg, gallery_model, query_loader, self.device, desc="gallery (old model)")
                gallery_labels = None
        else:
            gallery_features, gallery_labels = generate_feature_matrix(self.cfg, gallery_model, gallery_loader, self.device, desc="gallery (new model)")
        
        if cross_test_flag and old_emb_dim < emb_dim:
            query_features = query_features[:, :old_emb_dim]

        distmat = distance_metric(query_features, gallery_features)
        recall_list = self.cmc_optimized(distmat, query_labels, gallery_labels, log_writer, topk=self.cfg.EVAL.RECALL_RANK, desc="new2new")
        recall_1 = recall_list[0]
        if self.cfg.DATA.DATASET_TYPE == 'sop':
            mAP = 0
        else:
            mAP = self.mean_ap(distmat, query_labels, gallery_labels, log_writer)
        # all_cmc, mAP = self.cmc_map(distmat, query_labels, gallery_labels, log_writer, topk=self.cfg.EVAL.RECALL_RANK, desc="new2new")
        if mode == "old model self-test":
            self.omst_best_mAP = max(mAP, self.omst_best_mAP)
            self.omst_best_recall_1 = max(recall_1,self.omst_best_recall_1)
            logger.info(f"old model self-test best recall@1: {self.omst_best_recall_1:.4f}")
            log_writer.add_scalar("old model self-test recall", recall_1, epoch+1)
            logger.info(f"old model self-test best acc: {self.omst_best_mAP:.4f}")
            log_writer.add_scalar("old model self-test mAP", mAP, epoch+1)
            self.result_writer.write(epoch, self.omst_best_mAP)
        elif mode == "cross-test":
            self.ct_best_mAP = max(mAP, self.ct_best_mAP)
            self.ct_best_recall_1 = max(recall_1,self.ct_best_recall_1)
            logger.info(f"cross-test best recall@1: {self.ct_best_recall_1:.4f}")
            log_writer.add_scalar("cross-test recall", recall_1, epoch+1)
            logger.info(f"cross-test best acc: {self.ct_best_mAP:.4f}")
            log_writer.add_scalar("cross-test mAP", mAP, epoch+1)
        elif mode == "new model self-test":
            self.nmst_best_mAP = max(mAP, self.nmst_best_mAP)
            self.nmst_best_recall_1 = max(recall_1,self.nmst_best_recall_1)
            logger.info(f"new model self-test best recall@1: {self.nmst_best_recall_1:.4f}")
            log_writer.add_scalar("new model self-test recall", recall_1, epoch+1)
            logger.info(f"new model self-test best acc: {self.nmst_best_mAP:.4f}")
            log_writer.add_scalar("new model self-test mAP", mAP, epoch+1)



    def cmc_optimized(self,
        distmat: torch.Tensor,
        query_ids: Optional[torch.Tensor] = None,
        gallery_ids: Optional[torch.Tensor] = None,
        log_writer = None,
        topk: list = [1, 2, 4, 8],
        desc: str = "description"
    ) -> Tuple[float, float]:
        """Compute Cumulative Matching Characteristics metric.

        :param distmat: pairwise distance matrix between embeddings of gallery and query sets
        :param query_ids: labels for the query data. We're assuming query_ids and gallery_ids are the same.
        :param topk: parameter for top k retrieval
        :return: CMC top-1 and top-5 floats, as well as per-query top-1 and top-5 values.
        """
        distmat = copy.deepcopy(distmat)
        query_ids = copy.deepcopy(query_ids)
        if gallery_ids is None:
            distmat.fill_diagonal_(float("inf"))
            gallery_ids = query_ids
        else:
            gallery_ids = copy.deepcopy(gallery_ids)

        """
        # 这段代码和原来计算的recall代码原理一致，计算结果一致
        distmat_new_old_sorted, indices = torch.sort(distmat)# 默认dim=-1, 升序排列
        labels = gallery_ids.unsqueeze(dim=0).repeat(query_ids.shape[0], 1)
        sorted_labels = torch.gather(labels, 1, indices)
        top1_retrieval = sorted_labels[:, 0] == query_ids
        top5_retrieval = (
            #(sorted_labels[:, :topk] == query_ids.unsqueeze(1)).sum(dim=1).clamp(max=1)
            (sorted_labels[:, :4] == query_ids.unsqueeze(1)).sum(dim=1).clamp(max=1)
        )
        
        top1 = top1_retrieval.sum() / query_ids.shape[0]
        top5 = top5_retrieval.sum() / query_ids.shape[0]
        print(top1, top5)
        """

        num_querys = len(query_ids)
        idx = distmat.topk(k=topk[-1], dim=-1, largest=False)[1]
        recall_list = []
        for r in topk:
            correct = (gallery_ids[idx[:, 0:r]] == query_ids.unsqueeze(dim=-1)).any(dim=-1).float()
            recall_list.append((torch.sum(correct) / num_querys).item())

        for (k, _recall) in zip(topk, recall_list):
            logger.info(f"{desc}_Recall@{k} : {_recall:.2%}")
            if log_writer is not None:
                log_writer.add_scalar(f"{desc}_metric/Recall@{k}", _recall, self.iteration)
        return recall_list

    def mean_ap(self, distmat, query_ids=None, gallery_ids=None, log_writer=None, desc="description"):
        distmat = copy.deepcopy(distmat)
        m, n = distmat.shape
        # Fill up default values
        if query_ids is None:
            query_ids = np.arange(m)
        else:
            try:
                query_ids = query_ids.cpu().tolist()
            except AttributeError:
                pass

        if gallery_ids is None:
            distmat.fill_diagonal_(float("inf"))
            gallery_ids = copy.deepcopy(query_ids)
        else:
            try:
                gallery_ids = gallery_ids.cpu().tolist()
            except AttributeError:
                pass
        distmat = distmat.cpu().numpy().astype(np.float32)
        # Ensure numpy array
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        # Sort and find correct matches
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
        # Compute AP for each query
        aps = []
        for i in tqdm(range(m)):
            # Filter out the same img
            if list(query_ids) == list(gallery_ids):
                valid = (np.arange(n)[indices[i]] != np.arange(m)[i])
            else:
                valid = None
            y_true = matches[i, valid].reshape(-1)
            y_score = -distmat[i][indices[i]][valid].reshape(-1)
            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))
        if len(aps) == 0:
            raise RuntimeError("No valid query")
        mAP = np.mean(aps)
        logger.info(f"mean_AP : {mAP:.2%}")
        if log_writer is not None:
            log_writer.add_scalar(f"{desc}_metric/mean_AP", mAP, self.iteration)
        return mAP 

    
def cosine_distance_matrix(
    x: torch.Tensor, y: torch.Tensor, diag_only: bool = False
) -> torch.Tensor:
    """Get pair-wise cosine distances.

    :param x: A torch feature tensor with shape (n, d).
    :param y: A torch feature tensor with shape (n, d).
    :param diag_only: if True, only diagonal of distance matrix is computed and returned.
    :return: Distance tensor between features x and y with shape (n, n) if diag_only is False. Otherwise, elementwise
    distance tensor with shape (n,).
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    if diag_only:
        return 1.0 - torch.sum(x_norm * y_norm, dim=1)
    return 1.0 - x_norm @ y_norm.T


def l2_distance_matrix(
    x: torch.Tensor, y: torch.Tensor, diag_only: bool = False
) -> torch.Tensor:
    """Get pair-wise l2 distances.

    :param x: A torch feature tensor with shape (n, d).
    :param y: A torch feature tensor with shape (n, d).
    :param diag_only: if True, only diagonal of distance matrix is computed and returned.
    :return: Distance tensor between features x and y with shape (n, n) if diag_only is False. Otherwise, elementwise
    distance tensor with shape (n,).
    """
    if diag_only:
        return torch.norm(x - y, dim=1, p=2)
    return torch.cdist(x, y, p=2)

@torch.no_grad()
def generate_feature_matrix(
    cfg,
    model: Union[nn.Module, torch.jit.ScriptModule],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str,
    sparsity = None,
    parent_sparsity = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Feature Matrix
    :param model: Model to compute features.
    :param loader: Data loader to get gallery/query data.
    :param device: Device to use for computations.
    :param desc: description.
    :return: Three tensors gallery_features (n, d), query_features (n, d), labels (n,), where n is size of val dataset,
    and d is the embedding dimension.
    """

    model.eval()
    model.to(device)
    
    features = []
    labels = []

    for data, label, _ in tqdm(loader, total=len(loader), desc=desc):
        data, label = data.to(device), label.to(device)
        if cfg.SOLVER.TYPE == 'submodel_compatible':
            if cfg.SUB_MODEL.USE_SWITCHNET:
                model.apply(lambda m: setattr(m, 'width_mult', sparsity))
                feature,_,_ = model(data,return_feature=True)
            else:
                feature, _, _ = model(data, sparsity,mask=None,return_feature=True)
        else:
            feature, _, _ = model(data, return_feature=True)
        features.append(feature.squeeze().detach().cpu())
        labels.append(label)
    
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels


@torch.no_grad()
def generate_qg_feature_matrix(
    query_model: Union[nn.Module, torch.jit.ScriptModule],
    gallery_model: Union[nn.Module, torch.jit.ScriptModule],
    query_loader: torch.utils.data.DataLoader,
    gallery_loader: torch.utils.data.DataLoader,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Feature Matrix
    :param gallery_model: Model to compute gallery features.
    :param query_model: Model to compute query features.
    :param val_loader: Data loader to get gallery/query data.
    :param device: Device to use for computations.
    :param verbose: Whether to be verbose.
    :return: Three tensors gallery_features (n, d), query_features (n, d), labels (n,), where n is size of val dataset,
    and d is the embedding dimension.
    """

    gallery_model.eval()
    query_model.eval()

    gallery_model.to(device)
    query_model.to(device)

    gallery_features = []
    query_features = []
    gallery_labels = []
    query_labels = []

    for q_data, q_label in tqdm(query_loader, total=len(query_loader), desc="query"):
        q_data, q_label = q_data.to(device), q_label.to(device)
        query_feature, _, _ = query_model(q_data, return_feature=True)
        query_features.append(query_feature.squeeze().detach().cpu())
        query_labels.append(q_label)
    
    query_features = torch.cat(query_features)
    query_labels = torch.cat(query_labels)

    if gallery_loader is None:
        return query_features, None, query_labels, None
    else:
        for g_data, g_label in tqdm(gallery_loader, total=len(gallery_loader), desc="gallery"):
            g_data, g_label = g_data.to(device), g_label.to(device)
            gallery_feature, _, _ = gallery_model(g_data, return_feature=True)
            gallery_features.append(gallery_feature.squeeze().detach().cpu())
            gallery_labels.append(g_label)
        gallery_features = torch.cat(gallery_features)
        gallery_labels = torch.cat(gallery_labels)
    
        return query_features, gallery_features, query_labels, gallery_labels

"""
class Evaluator_Retrieval(Evaluator):
    def __init__(self,cfg, model, device, old_model=None):
        super(Evaluator_Retrieval, self).__init__()
        self.model = model
        self.old_model = old_model
        self.device = device
        self.cfg = cfg

    def evaluate(self, data_loader, query=None, gallery=None, metric=None):

        features, labels = extract_features(self.model, data_loader, self.device)

        if query is not None and gallery is not None:
            distmat = pairwise_distance(features, query, gallery, metric=metric)
            return evaluate_all(distmat, query=query, gallery=gallery)

        if self.old_model is not None:
            old_features, _ = extract_features(self.old_model, data_loader)
            distmat = pairwise_distance(features, old_features=old_features,
                                        query=None, gallery=None, metric=metric)
        else:
            distmat = pairwise_distance(features, old_features=None,
                                        query=None, gallery=None, metric=metric)
        return evaluate_all(distmat, query=labels, gallery=labels, cmc_topk=self.cfg.EVAL.RECALL_RANK)

@torch.no_grad()
def extract_features(model, data_loader,device, print_freq=1):
    
    batch_time = AverageMeter('Process Time', ':6.3f')
    data_time = AverageMeter('Test Date Time', ':6.3f')

    features = []
    labels = []

    end = time.time()
    for i, (imgs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end)
        imgs = imgs.to(device)
        outputs, embeddings_k, logis = model(imgs, return_feature=True)
        for output, pid in zip(outputs, targets):
            features.append(output)
            labels.append(pid)

        batch_time.update(time.time() - end)
        end = time.time()

    return features, labels


def pairwise_distance(features, old_features=None, query=None, gallery=None, metric=None):
    if old_features is None:
        old_features = features
    if query is None and gallery is None:
        n = len(features)
        #print(features.shape)
        x = torch.cat(features)
        y = torch.cat(old_features)
        x = x.view(n, -1)
        y = y.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
            y = metric.transform(y)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, n).t()
        dist = dist - 2 * torch.mm(x, y.t())
        return dist

    x = torch.cat([features[i].unsqueeze(0) for i, _ in enumerate(query)], 0)
    y = torch.cat([old_features[i].unsqueeze(0) for i, _ in enumerate(gallery)], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 cmc_topk=(1, 10, 100, 1000), compute_meanap=True):
    if query is not None and gallery is not None:
        query_ids = [label for label in query]
        gallery_ids = [label for label in gallery]

    # Compute mean AP
    if compute_meanap:
        T1 = time.time()
        meanap = mean_ap(distmat, query_ids=query_ids, gallery_ids=gallery_ids)
        T2 = time.time()
        print('time:%sms' % ((T2 - T1)*1000))
        meanap_fct = mean_ap_fct(distmat,query_ids)
        logger.info('Mean AP: {:4.1%}'.format(meanap))
        logger.info('Mean AP_FCL:{:4.1%}'.format(meanap_fct) )

    # Compute CMC scores
    cmc_scores = cmc(distmat, query_ids, gallery_ids, topk=1000,
                     single_gallery_shot=False,
                     first_match_break=True)

    for k in cmc_topk:
        logger.info('top-{:<4}{:12.1%}'
              .format(k, cmc_scores[k - 1]))

    # Use the cmc top-1 and top-5 score for validation criterion
    return cmc_scores[0], cmc_scores[4]
"""

class Evaluator_Submodel(Evaluator):
    def __init__(self, cfg, device, parent_model=None):
        super(Evaluator_Submodel, self).__init__()
        self.parent_model = parent_model
        self.device = device
        self.cfg = cfg
        self.result_writer = ResultWriter(os.path.join(cfg.OUTPUT_DIR, 'eval_result.txt'))
        self.sub_model_best_self_mAP =[0] * len(self.cfg.SUB_MODEL.SPARSITY)
        self.sub_model_best_cross_mAP =[0] * len(self.cfg.SUB_MODEL.SPARSITY)
        self.parent_model_best_mAP = 0.0
        self.sub_model_best_self_recall_1 = [0] * len(self.cfg.SUB_MODEL.SPARSITY)
        self.sub_model_best_cross_recall_1 = [0] * len(self.cfg.SUB_MODEL.SPARSITY)
        self.parent_model_best_recall_1 = 0.0
        self.best_p_down = 0.0
        self.sparsity = self.cfg.SUB_MODEL.SPARSITY if not self.cfg.SUB_MODEL.USE_SWITCHNET else self.cfg.SNET.WIDTH_MULT_LIST
        self.sub_model_num = len(self.sparsity) if not self.cfg.SUB_MODEL.USE_SWITCHNET else len(self.sparsity) -1 

    def _evaluate(self, _model, _old_model, sparsity, query_loader, gallery_loader, log_writer, epoch, mode="old model self-test",sub_model_idx=None,parent_sparsity=0):
        _model.eval()
        _old_model.eval()
        if mode in ["submodel self-test", "parent model self-test"]:
            query_model = _model
            gallery_model = _model
            logger.info("=> same-model test")
        elif mode == "submodel cross-test":
            query_model = _model
            gallery_model = _old_model
            logger.info("=> cross-model test")
        else:
            raise NotImplementedError
        distance_map = {"l2": l2_distance_matrix, "cosine": cosine_distance_matrix}
        
        feat_dim = _model.module.feat_dim if self.cfg.NUM_GPUS > 1 else _model.feat_dim# 获取backbone feature dim
        emb_dim = feat_dim if self.cfg.NEW_MODEL.PROJECTION_LAYERS < 0 else self.cfg.NEW_MODEL.PROJECTION_DIM# 判断是否有投影层
        cross_test_flag = False

        if _old_model is not None:
            old_feat_dim = _old_model.module.feat_dim if self.cfg.NUM_GPUS > 1 else _old_model.feat_dim# 获取old backbone feature dim
            old_emb_dim = old_feat_dim if self.cfg.MODEL.PROJECTION_LAYERS < 0 else self.cfg.MODEL.PROJECTION_DIM# 判断是否有投影层
            cross_test_flag = True

        distance_metric = distance_map.get(self.cfg.EVAL.DISTANCE_NAME.lower())
        logger.info('Generating Feature Matrix...')
        query_features, query_labels = generate_feature_matrix(self.cfg, query_model, query_loader, self.device, desc="query (new model)",sparsity=sparsity,parent_sparsity=parent_sparsity)
        if gallery_loader is None:
            if mode in ["submodel self-test", "parent model self-test"]:
                gallery_features = copy.deepcopy(query_features)
                gallery_labels = None
            elif mode == "submodel cross-test":
                gallery_features, _ = generate_feature_matrix(self.cfg, gallery_model, query_loader, self.device, desc="gallery (old model)",sparsity=parent_sparsity,parent_sparsity=parent_sparsity)
                gallery_labels = None
        else:
            if mode in ["submodel self-test", "parent model self-test"]:
                gallery_features, gallery_labels = generate_feature_matrix(self.cfg, gallery_model, gallery_loader, self.device, desc="gallery (new model)",sparsity=sparsity,parent_sparsity=parent_sparsity)
            else:
                gallery_features, gallery_labels = generate_feature_matrix(self.cfg, gallery_model, gallery_loader, self.device, desc="gallery (old model)",sparsity=parent_sparsity,parent_sparsity=parent_sparsity)

        if cross_test_flag and old_emb_dim < emb_dim:
            query_features = query_features[:, :old_emb_dim]

        distmat = distance_metric(query_features, gallery_features)
        recall_list = self.cmc_optimized(distmat, query_labels, gallery_labels, log_writer, topk=self.cfg.EVAL.RECALL_RANK, desc="new2new")
        recall_1 = recall_list[0]
        if self.cfg.DATA.DATASET_TYPE == 'sop':
            mAP = 0
        else:
            mAP = 0
            pass
            # mAP = self.mean_ap(distmat, query_labels, gallery_labels, log_writer)
        # all_cmc, mAP = self.cmc_map(distmat, query_labels, gallery_labels, log_writer, topk=self.cfg.EVAL.RECALL_RANK, desc="new2new")
        if mode == "parent model self-test":
            self.parent_model_best_mAP = max(mAP, self.parent_model_best_mAP)
            self.parent_model_best_recall_1 = max(recall_1,self.parent_model_best_recall_1)
            logger.info(f"parent model self-test best recall@1: {self.parent_model_best_recall_1:.4f}")
            log_writer.add_scalar("old model self-test recall", recall_1, epoch+1)
            logger.info(f"parent model self-test best acc: {self.parent_model_best_mAP:.4f}")
            log_writer.add_scalar("parent model self-test mAP", mAP, epoch+1)
        elif mode == "submodel self-test":
            idx = sub_model_idx
            self.sub_model_best_self_mAP[idx] = max(mAP, self.sub_model_best_self_mAP[idx])
            self.sub_model_best_self_recall_1[idx] = max(recall_1,self.sub_model_best_self_recall_1[idx])
            logger.info(f"submodel model {idx} sparsity {self.sparsity[idx]} best self recall@1: {self.sub_model_best_self_recall_1[idx]:.4f}")
            log_writer.add_scalar("submodel model self text recall", recall_1, epoch+1)
            logger.info(f"submodel model {idx} sparsity {self.sparsity[idx]} best self mAP: {self.sub_model_best_self_mAP[idx]:.4f}")
            log_writer.add_scalar("submodel model self text mAP", mAP, epoch+1)
        elif mode == "submodel cross-test":
            idx = sub_model_idx
            self.sub_model_best_cross_mAP[idx] = max(mAP, self.sub_model_best_cross_mAP[idx])
            self.sub_model_best_cross_recall_1[idx] = max(recall_1,self.sub_model_best_cross_recall_1[idx])
            logger.info(f"submodel model {idx} sparsity {self.sparsity[idx]} best cross recall@1: {self.sub_model_best_cross_recall_1[idx]:.4f}")
            log_writer.add_scalar("submodel model cross text recall", recall_1, epoch+1)
            logger.info(f"submodel model {idx} sparsity {self.sparsity[idx]} best cross mAP: {self.sub_model_best_cross_mAP[idx]:.4f}")
            log_writer.add_scalar("submodel model cross text mAP", mAP, epoch+1)
        return mAP


    def evaluate(self,parent_model,query_loader, gallery_loader, log_writer, epoch,sub_model_list=None,parent_sparsity=0):
        self.func = self.evaluate_gldv2 if self.cfg.DATA.NAME in ['lmdb-gldv2','retrieval-gldv2'] else self._evaluate

        if self.cfg.COMP_LOSS.TYPE in ['independent']:
            logger.info("=> parent-model test")
            self.func(_model=parent_model,_old_model=parent_model,sparsity=parent_sparsity,query_loader=query_loader,gallery_loader=gallery_loader,log_writer=log_writer,epoch=epoch,mode='parent model self-test',parent_sparsity=parent_sparsity)
            return
        else:
            map_list = {}
            logger.info("=> parent-model test")
            p_mAP = self.func(_model=parent_model,_old_model=parent_model,sparsity=parent_sparsity,query_loader=query_loader,gallery_loader=gallery_loader,log_writer=log_writer,epoch=epoch,mode='parent model self-test',parent_sparsity=parent_sparsity)
            map_list['parent'] = p_mAP
            if self.cfg.SUB_MODEL.ONLY_TEST_PARENT:
                return
            logger.info('=> sub-model test')
            if sub_model_list is not None:                
                for idx,sub_model in enumerate(sub_model_list):
                    if not self.cfg.SUB_MODEL.USE_SWITCHNET:
                        sparsity = self.cfg.SUB_MODEL.SPARSITY[idx]
                        sub_self_mAP = self.func(_model=sub_model,_old_model=sub_model,sparsity=sparsity,query_loader=query_loader,gallery_loader=gallery_loader,log_writer=log_writer,epoch=epoch,mode='submodel self-test',sub_model_idx=idx,parent_sparsity=parent_sparsity)                     
                        sub_cross_mAP = self.func(_model=sub_model,_old_model=parent_model,sparsity=sparsity,query_loader=query_loader,gallery_loader=gallery_loader,log_writer=log_writer,epoch=epoch,mode='submodel cross-test',sub_model_idx=idx,parent_sparsity=parent_sparsity)
                        map_list[sparsity] = [sub_self_mAP,sub_cross_mAP]
                    else:
                        sparsity = self.cfg.SNET.WIDTH_MULT_LIST[idx]
                        if sparsity != parent_sparsity:
                        # if sparsity <= float(parent_sparsity) and float(parent_sparsity)!=1.0:
                        #     sparsity = self.cfg.SNET.WIDTH_MULT_LIST[idx+1]
                            sub_self_mAP = self.func(_model=sub_model,_old_model=sub_model,sparsity=sparsity,query_loader=query_loader,gallery_loader=gallery_loader,log_writer=log_writer,epoch=epoch,mode='submodel self-test',sub_model_idx=idx,parent_sparsity=parent_sparsity)                     
                            sub_cross_mAP = self.func(_model=sub_model,_old_model=parent_model,sparsity=sparsity,query_loader=query_loader,gallery_loader=gallery_loader,log_writer=log_writer,epoch=epoch,mode='submodel cross-test',sub_model_idx=idx,parent_sparsity=parent_sparsity)
                            map_list[sparsity] = [sub_self_mAP,sub_cross_mAP]
            else:
                if self.cfg.SUB_MODEL.USE_SWITCHNET:
                    from ..models.subnetworks.slimmable_ops import SwitchableBatchNorm2d 
                    for idx,sparsity in enumerate(self.cfg.SNET.WIDTH_MULT_LIST):
                        if sparsity != parent_sparsity:
                            sub_self_mAP = self.func(_model=parent_model,_old_model=parent_model,sparsity=sparsity,query_loader=query_loader,gallery_loader=gallery_loader,log_writer=log_writer,epoch=epoch,mode='submodel self-test',sub_model_idx=idx)
                            sub_cross_mAP = self.func(_model=parent_model,_old_model=parent_model,sparsity=sparsity,query_loader=query_loader,gallery_loader=gallery_loader,log_writer=log_writer,epoch=epoch,mode='submodel cross-test',sub_model_idx=idx,parent_sparsity=parent_sparsity)
                            map_list[sparsity] = [sub_self_mAP,sub_cross_mAP] 
                else:                           
                    for idx,sparsity in enumerate(self.cfg.SUB_MODEL.SPARSITY):
                        sub_self_mAP = self.func(_model=parent_model,_old_model=parent_model,sparsity=sparsity,query_loader=query_loader,gallery_loader=gallery_loader,log_writer=log_writer,epoch=epoch,mode='submodel self-test',sub_model_idx=idx,parent_sparsity=parent_sparsity)
                        sub_cross_mAP = self.func(_model=parent_model,_old_model=parent_model,sparsity=sparsity,query_loader=query_loader,gallery_loader=gallery_loader,log_writer=log_writer,epoch=epoch,mode='submodel cross-test',sub_model_idx=idx,parent_sparsity=parent_sparsity)
                        map_list[sparsity] = [sub_self_mAP,sub_cross_mAP]                
            p_down = self.cal_average_matric(map_list)
            logger.info(f"submodel model p_down: {p_down:.4f}")
            self.best_p_down = max(self.best_p_down,p_down)
            logger.info(f"submodel model best p_down: {self.best_p_down:.4f}")
                
    def cal_average_matric(self,map_list):
        p_down = 0.0
        oo = map_list['parent']  
        for idx,sparsity in enumerate(self.sparsity):
            if sparsity != 1.0:
                nn = map_list[sparsity][0]
                no = map_list[sparsity][1]
                p_sub = no / (oo-nn)
                p_down += p_sub
        return p_down/len(self.sparsity)
    
    def cmc_optimized(self,
        distmat: torch.Tensor,
        query_ids: Optional[torch.Tensor] = None,
        gallery_ids: Optional[torch.Tensor] = None,
        log_writer = None,
        topk: list = [1, 2, 4, 8],
        desc: str = "description"
    ) -> Tuple[float, float]:
        """Compute Cumulative Matching Characteristics metric.

        :param distmat: pairwise distance matrix between embeddings of gallery and query sets
        :param query_ids: labels for the query data. We're assuming query_ids and gallery_ids are the same.
        :param topk: parameter for top k retrieval
        :return: CMC top-1 and top-5 floats, as well as per-query top-1 and top-5 values.
        """
        distmat = copy.deepcopy(distmat)
        query_ids = copy.deepcopy(query_ids)
        if gallery_ids is None:
            distmat.fill_diagonal_(float("inf"))
            gallery_ids = query_ids
        else:
            gallery_ids = copy.deepcopy(gallery_ids)

        num_querys = len(query_ids)
        idx = distmat.topk(k=topk[-1], dim=-1, largest=False)[1]
        recall_list = []
        for r in topk:
            correct = (gallery_ids[idx[:, 0:r]] == query_ids.unsqueeze(dim=-1)).any(dim=-1).float()
            recall_list.append((torch.sum(correct) / num_querys).item())

        for (k, _recall) in zip(topk, recall_list):
            logger.info(f"{desc}_Recall@{k} : {_recall:.2%}")
            if log_writer is not None:
                log_writer.add_scalar(f"{desc}_metric/Recall@{k}", _recall, self.iteration)
        return recall_list

    def mean_ap(self, distmat, query_ids=None, gallery_ids=None, log_writer=None, desc="description"):
        distmat = copy.deepcopy(distmat)
        m, n = distmat.shape
        # Fill up default values
        if query_ids is None:
            query_ids = np.arange(m)
        else:
            try:
                query_ids = query_ids.cpu().tolist()
            except AttributeError:
                pass

        if gallery_ids is None:
            distmat.fill_diagonal_(float("inf"))
            gallery_ids = copy.deepcopy(query_ids)
        else:
            try:
                gallery_ids = gallery_ids.cpu().tolist()
            except AttributeError:
                pass
        distmat = distmat.cpu().numpy().astype(np.float32)
        # Ensure numpy array
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        # Sort and find correct matches
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
        # Compute AP for each query
        aps = []
        for i in tqdm(range(m)):
            # Filter out the same img
            if list(query_ids) == list(gallery_ids):
                valid = (np.arange(n)[indices[i]] != np.arange(m)[i])
            else:
                valid = None
            y_true = matches[i, valid].reshape(-1)
            y_score = -distmat[i][indices[i]][valid].reshape(-1)
            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))
        if len(aps) == 0:
            raise RuntimeError("No valid query")
        mAP = np.mean(aps)
        logger.info(f"mean_AP : {mAP:.2%}")
        if log_writer is not None:
            log_writer.add_scalar(f"{desc}_metric/mean_AP", mAP, self.iteration)
        return mAP 
    
    def evaluate_gldv2(self, _model, _old_model, sparsity, query_loader, gallery_loader, log_writer, epoch, mode="parent model self-test",sub_model_idx=None,parent_sparsity=0):
        from .advBCT_evaluator import extract_features,concat_file,calculate_rank,calculate_mAP_gldv2,calculate_mAP_ijb_c,calculate_mAP_roxford_rparis
        _model.eval()
        _old_model.eval()
        if mode in ["submodel self-test", "parent model self-test"]:
            query_model = _model
            gallery_model = _model
            logger.info("=> same-model test")
        elif mode == "submodel cross-test":
            query_model = _model
            gallery_model = _old_model
            logger.info("=> cross-model test")
        query_gts = query_loader.dataset.query_gts
        dataset_name = self.cfg.EVAL.DATASET
        feat_dim = _model.module.feat_dim if self.cfg.NUM_GPUS > 1 else _model.feat_dim# 获取backbone feature dim
        emb_dim = feat_dim if self.cfg.NEW_MODEL.PROJECTION_LAYERS < 0 else self.cfg.NEW_MODEL.PROJECTION_DIM# 判断是否有投影层
        cross_test_flag = False

        if _old_model is not None:
            old_feat_dim = _old_model.module.feat_dim if self.cfg.NUM_GPUS > 1 else _old_model.feat_dim# 获取old backbone feature dim
            old_emb_dim = old_feat_dim if self.cfg.MODEL.PROJECTION_LAYERS < 0 else self.cfg.MODEL.PROJECTION_DIM# 判断是否有投影层
            cross_test_flag = True

        # print('query_loader', query_loader,gallery_loader, dist.get_rank())
        if self.cfg.NUM_GPUS<=1 or dist.get_rank() == 0:
            # extract query feat with new model
            extract_features(query_model, query_loader, 'q', logger, self.cfg, sparsity,parent_sparsity)
            # # extract gallery feat with old/new model
            if mode == "submodel cross-test":
                extract_features(gallery_model, gallery_loader, 'g', logger, self.cfg, parent_sparsity)  # use old_model to extract
            else:
                extract_features(gallery_model, gallery_loader, 'g', logger, self.cfg, sparsity,parent_sparsity)  # use old_model to extract
            # if dist.get_rank() == 0:
            # dist.barrier()
            # torch.cuda.empty_cache()  # empty gpu cache if using faiss gpu index

        mAP = 0.0
        if self.cfg.NUM_GPUS<=1 or dist.get_rank() == 0:
            logger.info("=> concat feat and label file")
            query_feats = concat_file(self.cfg.EVAL.SAVE_DIR, "feat_q",
                                    final_size=(len(query_loader.dataset), emb_dim))
            query_labels = concat_file(self.cfg.EVAL.SAVE_DIR, "label_q",
                                    final_size=(len(query_loader.dataset),))
            query_labels = query_labels.astype(np.int32)
            if mode in ['submodel self-test']:
                gallery_feats = concat_file(self.cfg.EVAL.SAVE_DIR, "feat_g",
                                            final_size=(len(gallery_loader.dataset), emb_dim))
                gallery_labels = concat_file(self.cfg.EVAL.SAVE_DIR, "label_g",
                                            final_size=(len(gallery_loader.dataset),))
                gallery_labels = gallery_labels.astype(np.int32)
            else:
                gallery_feats = concat_file(self.cfg.EVAL.SAVE_DIR, "feat_g_parent",
                                            final_size=(len(gallery_loader.dataset), emb_dim))
                gallery_labels = concat_file(self.cfg.EVAL.SAVE_DIR, "label_g_parent",
                                            final_size=(len(gallery_loader.dataset),))
                gallery_labels = gallery_labels.astype(np.int32)

            if cross_test_flag and old_emb_dim < emb_dim:
                print(query_feats.shape)
                query_feats = query_feats[:, :old_emb_dim]
                

            logger.info("=> calculate rank")
            if dataset_name == 'gldv2':
                ranked_gallery_indices = calculate_rank(logger, query_feats, gallery_feats, topk=100)
                logger.info("=> calculate mAP")
                mAP = calculate_mAP_gldv2(ranked_gallery_indices, query_gts[2], topk=100)
            elif dataset_name == 'ijb_c':
                ranked_gallery_indices = calculate_rank(logger, query_feats, gallery_feats, topk=100)
                logger.info("=> calculate mAP")
                mAP = calculate_mAP_ijb_c(ranked_gallery_indices, query_gts[2], topk=100)
            elif dataset_name == 'roxford5k' or dataset_name == 'rparis6k':
                ranked_gallery_indices = calculate_rank(logger, query_feats, gallery_feats, topk=gallery_feats.shape[0])
                logger.info("=> calculate mAP")
                mAP = calculate_mAP_roxford_rparis(logger, ranked_gallery_indices.transpose(), query_gts)
            else:
                raise ValueError
            # dist.barrier()
            # torch.cuda.empty_cache()
            print(f'mAP: {mAP}')
        if mode == "parent model self-test":
            self.parent_model_best_mAP = max(mAP, self.parent_model_best_mAP)
            logger.info(f"parent model self-test best acc: {self.parent_model_best_mAP:.4f}")
            log_writer.add_scalar("parent model self-test mAP", mAP, epoch+1)
        elif mode == "submodel self-test":
            idx = sub_model_idx
            self.sub_model_best_self_mAP[idx] = max(mAP, self.sub_model_best_self_mAP[idx])
            logger.info(f"submodel model {idx} sparsity {self.sparsity[idx]} best self mAP: {self.sub_model_best_self_mAP[idx]:.4f}")
            log_writer.add_scalar("submodel model self test mAP", mAP, epoch+1)
        elif mode == "submodel cross-test":
            idx = sub_model_idx
            self.sub_model_best_cross_mAP[idx] = max(mAP, self.sub_model_best_cross_mAP[idx])
            logger.info(f"submodel model {idx} sparsity {self.sparsity[idx]} best cross mAP1: {self.sub_model_best_cross_mAP[idx]:.4f}")
            log_writer.add_scalar("submodel model cross test mAP", mAP, epoch+1)
        return mAP