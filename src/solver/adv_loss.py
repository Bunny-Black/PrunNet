import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.losses import CrossBatchMemory, ContrastiveLoss
from pytorch_metric_learning.utils.module_with_records import ModuleWithRecords
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
__all__ = ['BackwardCompatibleLoss','UpgradeLoss']

def get_matches_and_diffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels

    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs

def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx

def mat_based_loss(mat, indices_tuple):
    a1, p, a2, n = indices_tuple
    pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
    pos_mask[a1, p] = 1
    neg_mask[a2, n] = 1
    return pos_mask, neg_mask

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


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    dist_m = dist_m.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist_m


def calculate_loss(feat_new, feat_old, feat_new_large, feat_old_large, targets_large,
                   masks, loss_type, temp, criterion,
                   loss_weight=1.0, topk_neg=-1):
    B, D = feat_new.shape
    indices_tuple = get_all_pairs_indices(targets_large)
    labels_idx = torch.arange(B) + torch.distributed.get_rank() * B
    if feat_new_large is None:
        feat_new_large = feat_new
        feat_old_large = feat_old

    if loss_type == 'contra':
        # 他这正样本对的定义是同一个样本用新、旧模型提取的特征
        logits_n2o_pos = torch.bmm(feat_new.view(B, 1, D), feat_old.view(B, D, 1))  # B*1
        logits_n2o_pos = torch.squeeze(logits_n2o_pos, 1)
        logits_n2o_neg = torch.mm(feat_new, feat_old_large.permute(1, 0))  # B*B
        logits_n2o_neg = logits_n2o_neg - masks * 1e9# 负无穷，排除所有同类别的负样本
        if topk_neg > 0:# 默认设置是10
            logits_n2o_neg = torch.topk(logits_n2o_neg, topk_neg, dim=1)[0]
        logits_all = torch.cat((logits_n2o_pos, logits_n2o_neg), 1)  # B*(1+k)
        logits_all /= temp

        labels_idx = torch.zeros(B).long().cuda()
        # 这个损失就是数学形式上一致，可以用这个代码接口来计算，用这个代码接口来计算，等效于：
        # 相当于有 k+1 个 class, logits_all 对应于B个样本在 k+1 个类别上的概率，然后labels_all给每个样本的类别标签是第0个类别，刚好logits_n2o也排在第0个位置，属于第0个类别
        loss = criterion(logits_all, labels_idx) * loss_weight
    elif loss_type in ['hot_refresh', 'bct_limit' ,'bct_limit_no_s2c']:
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
    elif loss_type == 'hard_info':
        new_new = torch.mm(feat_new_large, feat_new_large.t())
        new_old = torch.mm(feat_new_large, feat_old_large.t())
        mask_pos, mask_neg = mat_based_loss(new_new, indices_tuple)
        batch_size = new_new.shape[0]
        new_new[new_new==0] = torch.finfo(new_new.dtype).min
        pos_new_new = new_new*mask_pos
        neg_new_new = new_new*mask_neg

        new_old[new_old==0] = torch.finfo(new_old.dtype).min
        mask_pos = mask_pos + torch.eye(batch_size).to(mask_pos.device)
        mask_neg = mask_neg + torch.eye(batch_size).to(mask_neg.device)
        pos_new_old = new_old*mask_pos
        neg_new_old = new_old*mask_neg
        logits_all = []
        for i in range(batch_size):
            pos_item, neg_item = [], []
            pos_item.append(pos_new_new[i][pos_new_new[i] != 0])
            pos_item.append(pos_new_old[i][pos_new_old[i] != 0])
            neg_item.append(neg_new_new[i][neg_new_new[i] != 0])
            neg_item.append(neg_new_old[i][neg_new_old[i] != 0])
            pos_item = torch.cat(pos_item, dim=0)
            neg_item = torch.cat(neg_item, dim=0)
            pos = torch.topk(pos_item, largest=False, k=1)[0]
            neg = torch.topk(neg_item, largest=True, k=1791)[0]
            item = torch.cat([pos, neg], dim=0)
            logits_all.append(item.unsqueeze(0))
        logits_all = torch.cat(logits_all, 0)
        logits_all /= temp
        labels_idx = torch.zeros(batch_size).long().cuda()
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


class BackwardCompatibleLoss(nn.Module):
    def __init__(self, temp=0.01, margin=0.8, topk_neg=-1,
                 loss_type='contra', loss_weight=1.0, gather_all=True):
        super(BackwardCompatibleLoss, self).__init__()
        self.temperature = temp
        self.loss_weight = loss_weight
        self.topk_neg = topk_neg

        #   loss_type options:
        #   - contra
        #   - triplet
        #   - l2
        #   - hot_refresh (paper: "")
        #   - triplet_ract
        if loss_type in ['contra', 'hot_refresh', 'bct_limit','bct_limit_no_s2c', 'prototype_ce', 'elastic_bct_limit']:
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
        
        if self.loss_type == 'prototype_ce':# added by zikun
            #mem_size = feat_old.shape[0] # 类别数量
            #targets = targets.contiguous().view(-1, 1)
            #old_prototype_targets = torch.arange(mem_size).to(targets.device).contiguous().view(-1, 1)
            #masks = torch.eq(targets, old_prototype_targets.T).to(targets.device)
            logits_n2o_all = torch.mm(feat, feat_old.T)
            #logits_n2o_all = logits_n2o_all / self.temperature
            # 去掉前面代码里面的归一化训练了10多个epoch, cross-test mAP就还是2%
            loss_comp = self.criterion(logits_n2o_all, targets)
            return loss_comp
        else:
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
                feat_large, feat_old_large, targets_large = None, None, None
                masks = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())

        # compute loss
        loss_comp = calculate_loss(feat, feat_old, feat_large, feat_old_large, targets_large, masks, self.loss_type, self.temperature,
                                   self.criterion, self.loss_weight, self.topk_neg)
        return loss_comp


class UpgradeLoss(ModuleWithRecords):
    def __init__(self, loss, embedding_size=256, memory_size=500,miner=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.embedding_memory = torch.zeros(self.memory_size, self.embedding_size)
        self.label_memory = torch.zeros(self.memory_size).long()
        self.has_been_filled = False
        self.queue_idx = 0
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "memory_size", "queue_idx"], is_stat=False
        )
        self.loss = loss

    def forward(self, embeddings, labels, indices_tuple=None, enqueue_idx=None):
        if enqueue_idx is not None:
            assert len(enqueue_idx) <= len(self.embedding_memory)
            assert len(enqueue_idx) < len(embeddings)
        else:
            assert len(embeddings) <= len(self.embedding_memory)
        self.reset_stats()
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        self.embedding_memory = c_f.to_device(
            self.embedding_memory, device=device, dtype=embeddings.dtype
        )
        self.label_memory = c_f.to_device(
            self.label_memory, device=device, dtype=labels.dtype
        )
        do_remove_self_comparisons = False
        if enqueue_idx is not None:
            mask = torch.zeros(len(embeddings), device=device, dtype=torch.bool)
            mask[enqueue_idx] = True
            emb_for_queue = embeddings[mask]
            labels_for_queue = labels[mask]
            embeddings = embeddings[~mask]
            labels = labels[~mask]
            # do_remove_self_comparisons = False
        else:
            emb_for_queue = embeddings
            labels_for_queue = labels
            # do_remove_self_comparisons = True

        queue_batch_size = len(emb_for_queue)
        self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)

        if not self.has_been_filled:
            E_mem = self.embedding_memory[: self.queue_idx]
            L_mem = self.label_memory[: self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory

        combined_embeddings = torch.cat([embeddings, E_mem], dim=0)
        combined_labels = torch.cat([labels, L_mem], dim=0)

        # combined_embeddings_normed = F.normalize(combined_embeddings, dim=1, p=2)
        combined_embeddings_normed = combined_embeddings

        labels_set = set(combined_labels.detach().cpu().tolist())
        center_embeddings = torch.zeros([len(labels_set), self.embedding_size])
        center_embeddings = c_f.to_device(
            center_embeddings, device=device, dtype=embeddings.dtype
        )
        center_labels = torch.zeros(len(labels_set))
        center_labels = c_f.to_device(
            center_labels, device=device, dtype=labels.dtype
        )

        for i,l in enumerate(labels_set):
            center_i_features = combined_embeddings_normed[combined_labels==l]
            center = torch.mean(center_i_features,dim=0)
            # center = F.normalize(center[None,:],p=2)
            center_embeddings[i] = center
            center_labels[i] = i

        indices_tuple = self.create_indices_tuple(
            center_embeddings.shape[0],
            center_embeddings,
            center_labels,
            indices_tuple,
            do_remove_self_comparisons
        )
        loss = self.loss(center_embeddings, center_labels, indices_tuple)
        return loss

    def create_indices_tuple(
        self,
        batch_size,
        embeddings,
        labels,
        input_indices_tuple,
        do_remove_self_comparisons,
    ):
        indices_tuple = lmu.get_all_pairs_indices(labels, labels)

        if do_remove_self_comparisons:
            indices_tuple = self.remove_self_comparisons(indices_tuple)

        # indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)

        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(
                    input_indices_tuple, labels
                )
            indices_tuple = tuple(
                [
                    torch.cat([x, c_f.to_device(y, x)], dim=0)
                    for x, y in zip(indices_tuple, input_indices_tuple)
                ]
            )

        return indices_tuple

    def add_to_memory(self, embeddings, labels, batch_size):
        self.curr_batch_idx = (
            torch.arange(
                self.queue_idx, self.queue_idx + batch_size, device=labels.device
            )
            % self.memory_size
        )
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
        self.label_memory[self.curr_batch_idx] = labels.detach()
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True

    def remove_self_comparisons(self, indices_tuple):
        # remove self-comparisons
        assert len(indices_tuple) in [3, 4]
        s, e = self.curr_batch_idx[0], self.curr_batch_idx[-1]
        if len(indices_tuple) == 3:
            a, p, n = indices_tuple
            keep_mask = self.not_self_comparisons(a, p, s, e)
            a = a[keep_mask]
            p = p[keep_mask]
            n = n[keep_mask]
            assert len(a) == len(p) == len(n)
            return a, p, n
        elif len(indices_tuple) == 4:
            a1, p, a2, n = indices_tuple
            keep_mask = self.not_self_comparisons(a1, p, s, e)
            a1 = a1[keep_mask]
            p = p[keep_mask]
            assert len(a1) == len(p)
            assert len(a2) == len(n)
            return a1, p, a2, n

    # a: anchors
    # p: positives
    # s: curr batch start idx in queue
    # e: curr batch end idx in queue
    def not_self_comparisons(self, a, p, s, e):
        curr_batch = torch.any(p.unsqueeze(1) == self.curr_batch_idx, dim=1)
        a_c = a[curr_batch]
        p_c = p[curr_batch]
        p_c -= s
        if e <= s:
            p_c[p_c <= e - s] += self.memory_size
        without_self_comparisons = curr_batch.clone()
        without_self_comparisons[torch.where(curr_batch)[0][a_c == p_c]] = False
        return without_self_comparisons | ~curr_batch


class UpgradeCenterLoss(ModuleWithRecords):
    def __init__(self, loss, embedding_size=256, num_class=500,device_id=0,device_num=8,dataset_len=0):
        super().__init__()
        self.num_class = num_class
        self.embedding_size = embedding_size
        cur_center_num = num_class // device_num
        self.cur_center_num = cur_center_num
        self.device_id = device_id
        self.device_num = device_num
        self.dataset_len = dataset_len
        self.counter = 0
        if device_id < device_num-1:
            self.class_center = torch.zeros([cur_center_num,embedding_size]).float().to(device_id)
            self.class_past_num = torch.zeros([cur_center_num]).long().to(device_id)
            self.class_curr_id = range(device_id*cur_center_num,cur_center_num*(device_id+1))
        else:
            self.class_center = torch.zeros([cur_center_num+num_class-cur_center_num*device_num, embedding_size]).float().to(device_id)
            self.class_past_num = torch.zeros([cur_center_num+num_class-cur_center_num*device_num]).long().to(device_id)
            self.class_curr_id = range(device_id * cur_center_num, num_class)
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "num_class","cur_center_num"], is_stat=False
        )
        self.margin = 1
        self.loss = loss

    def shift_index(self,l):
        return l-self.cur_center_num*self.device_id

    def reset_center(self):
        if self.device_id < self.device_num-1:
            self.class_center = torch.zeros([self.cur_center_num,self.embedding_size]).float().to(self.device_id)
            self.class_past_num = torch.zeros([self.cur_center_num]).long().to(self.device_id)
            self.class_curr_id = range(self.device_id*self.cur_center_num,self.cur_center_num*(self.device_id+1))
        else:
            self.class_center = torch.zeros([self.cur_center_num+self.num_class-self.cur_center_num*self.device_num, self.embedding_size]).float().to(self.device_id)
            self.class_past_num = torch.zeros([self.cur_center_num+self.num_class-self.cur_center_num*self.device_num]).long().to(self.device_id)
            self.class_curr_id = range(self.device_id * self.cur_center_num, self.num_class)
        self.counter = 0

    def forward(self, embeddings, labels,indices_tuple=None,reset=False):
        device = embeddings.device
        self.counter += 1
        self.class_center = c_f.to_device(
            self.class_center, device=device, dtype=embeddings.dtype
        )
        for l in set(labels.detach().cpu().tolist()):
            if l not in self.class_curr_id:
                continue
            feat_l = embeddings[labels==l].detach()
            feat_l = F.normalize(feat_l)
            l = self.shift_index(l)
            curr_num = feat_l.shape[0]+self.class_past_num[l]
            self.class_center[l] = self.class_center[l]*(self.class_past_num[l]/curr_num) + (torch.sum(feat_l,dim=0)/curr_num)
            self.class_center[l] = F.normalize(self.class_center[l][None,:])
            self.class_past_num[l] = curr_num
        valid_label_idx = self.class_past_num != 0
        center_labels = c_f.to_device(
            torch.arange(0,sum(valid_label_idx)), device=device, dtype=labels.dtype
        )

        dist = torch.mm(self.class_center[valid_label_idx],self.class_center[valid_label_idx].t())
        dist_up = torch.triu(dist,diagonal=1)
        dist_up = 1-dist_up[dist_up != 0]
        dist_up = self.margin-dist_up
        dist_up = dist_up[dist_up > 0]
        loss = dist_up.mean()
        # loss = self.loss(self.class_center[valid_label_idx], center_labels, indices_tuple)
        if self.counter == self.dataset_len:
            print('reset upgrade class center')
            self.reset_center()
        return loss

    def create_indices_tuple(
        self,
        batch_size,
        embeddings,
        labels,
        input_indices_tuple
    ):
        indices_tuple = lmu.get_all_pairs_indices(labels, labels)

        # if do_remove_self_comparisons:
        #     indices_tuple = self.remove_self_comparisons(indices_tuple)

        # indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)

        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(
                    input_indices_tuple, labels
                )
            indices_tuple = tuple(
                [
                    torch.cat([x, c_f.to_device(y, x)], dim=0)
                    for x, y in zip(indices_tuple, input_indices_tuple)
                ]
            )

        return indices_tuple

class UpgradeCenterPartialLoss(CrossBatchMemory):
    def __init__(self, loss, embedding_size=256, num_class=500,device_id=0,device_num=8,dataset_len=0):
        super().__init__(None,embedding_size=embedding_size)
        self.num_class = num_class
        self.embedding_size = embedding_size
        self.device_id = device_id
        self.device_num = device_num
        self.dataset_len = dataset_len
        self.counter = 0
        self.class_center = torch.zeros([num_class,embedding_size]).float()
        self.class_past_num = torch.zeros([num_class]).long()
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "num_class","cur_center_num"], is_stat=False
        )
        self.margin = 1
        self.loss = loss


    def reset_center(self):
        self.class_center = torch.zeros([self.num_class, self.embedding_size]).float()
        self.class_past_num = torch.zeros([self.num_class]).long()
        self.counter = 0

    def forward(self, embeddings, labels,indices_tuple=None, enqueue_idx=None):
        device = embeddings.device
        self.counter += 1

        curr_embedding = torch.zeros([len(set(labels.detach().cpu().tolist())),self.embedding_size]).float().to(device)
        curr_class = torch.zeros([len(set(labels.detach().cpu().tolist()))]).long().to(device)

        for i,l in enumerate(set(labels.detach().cpu().tolist())):
            feat_l = embeddings[labels==l].detach()
            feat_l = F.normalize(feat_l)
            # new instance num of class l
            new_num = feat_l.shape[0]+self.class_past_num[l]
            # fetch the center of class l
            center = self.class_center[l].to(device)
            # fetch history class num
            past_num = self.class_past_num[l]
            # update the center of class l
            curr_embedding[i] = center*(past_num/new_num) + (torch.sum(feat_l,dim=0)/new_num) # (center*past_num+sum(feat_l))/new_num
            curr_embedding[i] = F.normalize(curr_embedding[i][None,:])
            self.class_past_num[l] = new_num
            curr_class[i] = i

        dist = torch.mm(curr_embedding,curr_embedding.t())
        dist_up = torch.triu(dist,diagonal=1)
        dist_up = 1-dist_up[dist_up != 0]
        dist_up = self.margin-dist_up
        dist_up = dist_up[dist_up > 0]
        loss = dist_up.mean()
        # loss = self.loss(self.class_center[valid_label_idx], center_labels, indices_tuple)
        if self.counter == self.dataset_len:
            print('reset upgrade class center')
            self.reset_center()
        return loss

class CustomContrastiveLoss(ContrastiveLoss):
    """Training Vision Transformers for Image Retrieval equation (1)
    """
    def __init__(self, cfg = None, distance=CosineSimilarity(),**kwargs):
        super(CustomContrastiveLoss, self).__init__(
            pos_margin=cfg.LOSS.LOSS_POS_MARGIN,
            neg_margin=cfg.LOSS.LOSS_NEG_MARGIN,
            distance=distance,
            **kwargs)
        
class CE_WITH_SMOOTH(nn.Module):
    def __init__(self, cfg):
        super(CE_WITH_SMOOTH, self).__init__()
        self.cfg = cfg
        
    def forward(self, pred_class_outputs, gt_classes, eps, alpha=0.2,cfg=None,parent_cls=None):
        num_classes = pred_class_outputs.size(1)

        if eps >= 0:
            smooth_param = eps
        else:
            # Adaptive label smooth regularization
            soft_label = F.softmax(pred_class_outputs, dim=1)
            smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

        log_probs = F.log_softmax(pred_class_outputs, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (num_classes - 1)
            targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

        loss = (-targets * log_probs).sum(dim=1)

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
        if cfg and cfg.SUBNET.add_hard_scale:
            _loss = loss.detach()
            mean = _loss.mean()
            # std = compatible_loss.std()
            _loss_norm = (_loss - mean) / mean
            hard_sacle = 1 + _loss_norm
            loss = loss * hard_sacle
        if cfg and cfg.SUBNET.add_kl_scale and parent_cls is not None:
            cls_score_norm = F.softmax(parent_cls, dim=-1)
            cls_score_sub_norm = F.softmax(pred_class_outputs, dim=-1)
            p = cls_score_norm.detach() + 1e-10
            q = cls_score_sub_norm.detach() + 1e-10
            kl_divergence = torch.sum(p * torch.log(p / q), dim=-1)
            mean_kl = torch.mean(kl_divergence)
            kl_scale = 1 + (kl_divergence - mean_kl)
            loss = loss * kl_scale
        

        loss = loss.sum() / non_zero_cnt

        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



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
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_feature = contrast_feature = features# all view
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数
        # mask-out self-contrast cases, 四个输入分别是 input, dim, index, src
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # compute log_prob, 按照 Supervised Contrastive Learning Equation (2)
        exp_logits = torch.exp(logits) * logits_mask
        # print(exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        # print(mean_log_prob_pos)
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
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
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
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数

        # mask-out self-contrast cases, 在compatible learning的时候有意义吗? 同一个样本，新、旧feature有必要自己跟自己比较吗？
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        logits_mask = torch.ones_like(mask).to(device)# 自己和自己比也应该保留
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
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数
        #raise Exception
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
        # NCCL那篇论文从公式上看，是有old memory
        if not withnew:
            return self._cal_loss_wo_new(features, labels, old_features, old_labels)
        elif withnew and (not withnewxbm):# 兼容没有old-xbm的情况
            return self._cal_loss_wi_new(features, labels, old_features, old_labels, img_ids, old_img_ids)

class CompatiblePrototypeLoss(nn.Module):
    def __init__(self, cfg, temperature=0.07, base_temperature=0.07) -> None:
        super(CompatiblePrototypeLoss, self).__init__()
        self.cfg = cfg
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, old_prototype, new_prototype=None):
        device = features.device
        batch_size = features.shape[0]
        if new_prototype is None:
            mem_size = old_prototype.shape[0]
        else:
            mem_size = old_prototype.shape[0] + new_prototype.shape[0] # 类别数量
        
        labels = labels.contiguous().view(-1, 1)
        prototype_labels = torch.arange(mem_size).to(labels.device).contiguous().view(-1, 1)
        mask = torch.eq(labels, prototype_labels.T).float().to(device)
        if self.cfg.COMP_LOSS.NEW_CLASS_PROTOTYPE == 'null':
            effective_sample = mask.sum(dim=1) > 0
            effective_index = (effective_sample.int() * torch.arange(batch_size).to(device))[effective_sample]
            mask = torch.index_select(mask, dim=0, index=effective_index)
        
        # compute logits
        prototype = old_prototype if new_prototype is None else torch.cat([old_prototype, new_prototype], dim=0)
        anchor_dot_contrast = torch.div(torch.matmul(features, prototype.T), self.temperature)
        if self.cfg.COMP_LOSS.NEW_CLASS_PROTOTYPE == 'null':
            anchor_dot_contrast = torch.index_select(anchor_dot_contrast, dim=0, index=effective_index)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if self.cfg.COMP_LOSS.NEW_CLASS_PROTOTYPE == 'null':
            loss = loss.view(1, len(effective_index)).mean()
        else:
            loss = loss.view(1, batch_size).mean()
        if torch.isnan(loss):
            raise Exception
        return loss
    
class CompatibleElasticPrototypeLoss(nn.Module):
    def __init__(self, cfg, temperature=0.07, base_temperature=0.07) -> None:
        super(CompatibleElasticPrototypeLoss, self).__init__()
        self.cfg = cfg
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.scale = cfg.COMP_LOSS.ELASTIC_SCALE
        self.scale_new = cfg.COMP_LOSS.NEW_SCALE

    def forward(self, features, labels, old_prototype, top_similarities=None, top_similar_ids=None, support_prototypes=None, 
                new_prototype=None, o2n_top_similarities_all=None,new_prototypes_support=None,disturbed_old_prototype=None, fix_disturb=False,add_n2o_disturb=False):
        device = features.device
        batch_size = features.shape[0]
        mem_size = old_prototype.shape[0] # 类别数量

        labels = labels.contiguous().view(-1, 1)
        old_prototype_labels = torch.arange(mem_size).to(labels.device).contiguous().view(-1, 1)
        # if new_prototype is not None:
        #     o2n_top_similarities_all,new_prototypes_support = self.analyze_new_prototype(old_prototype, new_prototype)
        # else:
        #     o2n_top_similarities_all = None
        #     new_prototypes_support = None
        mask = torch.eq(labels, old_prototype_labels.T).float().to(device)# (B, Class)
        if not fix_disturb:
            disturbed_old_prototype = self.add_disturb_v0(old_prototype, top_similarities, top_similar_ids, support_prototypes, o2n_top_similarities_all,new_prototypes_support)
            if add_n2o_disturb:
                disturbed_old_prototype = self.add_disturb_n20(disturbed_old_prototype,o2n_top_similarities_all,new_prototypes_support)
        # disturbed_old_prototype = self.add_disturb_v1(old_prototype, top_similarities, top_similar_ids, support_prototypes, new_prototype)
        # 正样本prototype有扰动，负样本prototype不需要扰动？同一个prototype，对于不同的样本，可能是正样本，也可能是负样本
        similarities_wo_disturb = torch.matmul(features, old_prototype.T)
        similarities_wi_disturb = torch.matmul(features, disturbed_old_prototype.T)
        similarities = similarities_wo_disturb * (1 - mask) + similarities_wi_disturb * mask# (B, Class)

        # compute logits
        anchor_dot_contrast = torch.div(similarities, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()# 都是负数

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss
        
    def analyze_new_prototype(self, old_prototypes, new_prototypes):
        # old_prototype, new_prototype 都是经过归一化的
        o2n_top_similarities_all, o2n_top_similar_ids_all, o2n_support_prototypes = [], [], []
        new_prototypes_topk = []
        for label, old_ptt in enumerate(old_prototypes):
            #print(label, center_norm.shape)
            similarities = torch.matmul(old_ptt.unsqueeze(0), new_prototypes.T).squeeze()
            # similarities = torch.cat([similarities[:label], similarities[label+1:]], dim=0)# 去掉正样本对相似度，这里只考虑负样本对
            top_similarities, top_similar_ids = torch.topk(similarities, k=self.cfg.COMP_LOSS.TOPK_FOR_SUPPORT, dim=0, largest=True)
            o2n_top_similarities_all.append(top_similarities)
            new_prototypes_topk.append(new_prototypes[top_similar_ids])
        o2n_top_similarities_all = torch.stack(o2n_top_similarities_all, dim=0)
        new_prototypes_support = torch.stack(new_prototypes_topk, dim=0)
        return o2n_top_similarities_all,new_prototypes_support
        #top_similar_ids_all = torch.stack(top_similar_ids_all, dim=0)
        #support_prototypes = torch.stack(support_prototypes, dim=0)# (Class, topk ,C)
        #return top_similarities_all, top_similar_ids_all, support_prototypes

    def add_disturb_v1(self, old_prototype, support_similarities, support_ids, support_prototypes, new_prototype):
        # old_prototype: (class, C), support_similarities: (class, topk), support_prototypes: (class, topk, C)
        #print('norm', torch.norm((support_similarities.unsqueeze(-1) * (old_prototype.unsqueeze(1) - support_prototypes)).sum(dim=1), dim=-1, p=2))
        #print("support_similarities", support_similarities.unsqueeze(-1).sum(1))
        # 这个扰动属于固定值扰动，因为用来计算扰动的几个变量就没变过，而且也和具体的样本没有关系，这个设计有点low
        new_support_prototypes, o2n_support_similarities = [], []
        for i, sid in enumerate(support_ids):
            new_support_prototypes.append(new_prototype[sid])
            o2n_support_similarity = torch.matmul(old_prototype[i], new_prototype[sid].T).squeeze()
            o2n_support_similarities.append(o2n_support_similarity)
        new_support_prototypes = torch.stack(new_support_prototypes, dim=0)
        o2n_support_similarities = torch.stack(o2n_support_similarities, dim=0)
        #print(o2n_support_similarities.shape, support_similarities.shape)
        weights = support_similarities + 0.2 * (support_similarities - o2n_support_similarities)
        disturbs = (weights.unsqueeze(-1) * (old_prototype.unsqueeze(1) - support_prototypes)).sum(dim=1) / (weights.unsqueeze(-1).sum(1)+1e-8)
        disturb_prototype = old_prototype + self.scale * disturbs
        return disturb_prototype

    def add_disturb_v0(self, old_prototype, support_similarities, support_ids, support_prototypes, o2n_top_similarities,new_prototype = None):
        # old_prototype: (class, C), support_similarities: (class, topk), support_prototypes: (class, topk, C)
        #print('norm', torch.norm((support_similarities.unsqueeze(-1) * (old_prototype.unsqueeze(1) - support_prototypes)).sum(dim=1), dim=-1, p=2))
        #print("support_similarities", support_similarities.unsqueeze(-1).sum(1))
        # 这个扰动属于固定值扰动，因为用来计算扰动的几个变量就没变过，而且也和具体的样本没有关系，这个设计有点low
        disturbs = self.scale * (support_similarities.unsqueeze(-1) * (old_prototype.unsqueeze(1) - support_prototypes)).sum(dim=1) / (support_similarities.unsqueeze(-1).sum(1)+1e-8)
        if o2n_top_similarities is not None:
            # self.scale_n2o = self.scale * (1 - o2n_top_similarities.mean(dim=1)).unsqueeze(-1)
            disturbs_n2o = (o2n_top_similarities.unsqueeze(-1) * (old_prototype.unsqueeze(1) - new_prototype)).sum(dim=1) / (o2n_top_similarities.unsqueeze(-1).sum(1)+1e-8)    
            disturbs += self.scale_new * disturbs_n2o    
        if self.cfg.COMP_LOSS.NORM_PROTOTYPE:
            disturb_prototype = F.normalize(old_prototype + self.scale * disturbs, dim=1)# TODO: check 增加了一个归一化操作
        else:
            disturb_prototype = old_prototype + disturbs # TODO: check 增加了一个归一化操作
        """
        diff = torch.bmm(old_prototype.unsqueeze(1), disturb_prototype.unsqueeze(-1))
        theta = torch.acos(diff)
        #disturb_prototype = 1.75 * old_prototype# 第一次尝试这个超参数是1.25
        print('diff', diff.squeeze().cpu().tolist()[:100])
        print('theta', theta.squeeze().cpu().tolist()[:100])
        #disturb_prototype2 = old_prototype + self.scale * disturbs
        #diff2 = torch.bmm(old_prototype.unsqueeze(1), disturb_prototype2.unsqueeze(-1))
        #print('diff2', diff2.squeeze().cpu().tolist()[:100])
        #raise Exception
        """
        return disturb_prototype

    
    def add_disturb_n20(self, old_prototype, o2n_top_similarities,new_prototype = None):
        assert o2n_top_similarities is not None ,'o2n_top_similarities does not exist.'
            # self.scale_n2o = self.scale * (1 - o2n_top_similarities.mean(dim=1)).unsqueeze(-1)
        disturbs_n2o = (o2n_top_similarities.unsqueeze(-1) * (old_prototype.unsqueeze(1) - new_prototype)).sum(dim=1) / (o2n_top_similarities.unsqueeze(-1).sum(1)+1e-8)    
        if self.cfg.COMP_LOSS.NORM_PROTOTYPE:
            disturb_prototype = F.normalize(old_prototype + self.scale * disturbs_n2o, dim=1)# TODO: check 增加了一个归一化操作
        else:
            disturb_prototype = old_prototype + self.scale * disturbs_n2o # TODO: check 增加了一个归一化操作
        return disturb_prototype