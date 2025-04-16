import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import copy
import os
import json
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import random
from torch.utils.data import RandomSampler
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data.distributed import DistributedSampler

# 创建命令行参数解析器
# parser = argparse.ArgumentParser()
# parser.add_argument('--threshold', type=float, help='设置阈值')
# parser.add_argument('--batchsize', type=int, help='设置batchsize')
# parser.add_argument('--epoch', type=int, help='设置epoch')

# 解析命令行参数
# args = parser.parse_args()

# # 获取 threshold 的值
# threshold = args.threshold
# batchsize = int(args.batchsize)
# num_epoch = args.epoch
# 设置相似度阈值
# threshold = 0.6

def calculate_cosine_similarity_matrix(vectors):
    # 将向量转换为NumPy数组
    vectors_array = np.array(vectors)
    
    # 计算余弦相似度矩阵
    cosine_sim_matrix = cosine_similarity(vectors_array)
    
    return cosine_sim_matrix


def save_centers(centers):
    data_dict = {}

    feat = np.load('/data/sunyushuai/FCL_ViT/test_results/resnet_lmdb_gldv2_allmini_softmax_lmdb_ep06/lmdb-gldv2_142772_feat.npy')
    label = np.load('/data/sunyushuai/FCL_ViT/test_results/resnet_lmdb_gldv2_allmini_softmax_lmdb_ep06/lmdb-gldv2_142772_label.npy')
    print('feat, label', feat.shape, label.shape)
    feat_dict = defaultdict(list)
    for i in range(len(label)):
        feat[i] = feat[i] / np.linalg.norm(feat[i])
        feat_dict[int(label[i])].append(feat[i])

        for i, k in enumerate(feat_dict.keys()):
            # center = np.asarray(feat_dict[k]).mean(0)# 未归一化的类别中心，但是用于计算类别中心的都归一化过
            center = centers[i].detach().numpy()
            maxtmp = 0
            radius = []
            for f in feat_dict[k]:
                diff = f - center# 这里f是归一化的，但是center是未归一化的，所以使用的减法，计算l2distance?
                tmp = np.linalg.norm(diff)
                maxtmp = max(tmp, maxtmp)
                radius.append(tmp.item())

            radius = sorted(radius)# 对每个样本到其类别中心的距离，升序排列
            # 1.5IQR
            radius_new = []
            maxtmp = 0
            if len(radius) >= 4:# 排除一些外点的影响
                nu = len(radius)
                q3, q1 = radius[int(3 * nu / 4)], radius[int(nu / 4)]
                IQR = q3 - q1# IQR: 四分位距，interquatile range, IQR
                for r in radius:
                    if r < q1 - 1.5 * IQR or r > q3 + 1.5 * IQR:
                        continue
                    maxtmp = max(r, maxtmp)
                    radius_new.append(r)
            else:
                for r in radius:
                    maxtmp = max(r, maxtmp)
                    radius_new.append(r)
            if len(radius_new) == 0:
                radius_new = radius

            data_dict[k] = {'center': center.tolist(), 'radius': radius_new}
            # print(i, len(feat_dict), k, len(radius), radius, maxtmp)

    with open('perturbed_prototype_meta_radius_centernorm.json', 'w') as fw:
        json.dump(data_dict, fw)

def load_old_centers():
    path = '/data/sunyushuai/FCL_ViT/test_results/resnet_lmdb_gldv2_allmini_softmax_lmdb_ep06/lmdb-gldv2_470369_meta_radius_centernorm.json'
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        old_meta = json.load(f)
    centers = []
    for i in range(24393):
        centers.append(old_meta[str(i)]['center'])
    centers = torch.tensor(centers).float()
    print(f'centers shape: {centers.shape}')
    return centers


# perturbed_centers = np.load('/data/sunyushuai/perturbed_old_centers.npy')
# cosine_sim_matrix = calculate_cosine_similarity_matrix(old_centers)
# positive_values = cosine_sim_matrix[cosine_sim_matrix > 0]
# print(positive_values)
# mean = np.mean(positive_values)
# std = np.std(positive_values)

# print("均值:", mean)
# print("标准差:", std)
# exit()


# 统计相似度大于阈值的样本对数量
# num_similar_pairs = np.sum(cosine_sim_matrix > threshold)
# print(num_similar_pairs)
# exit()

# centers = copy.deepcopy(old_centers)


def train_loop(perturbation, old_centers, batch_size, num_epochs, learning_rate,threshold, print_interval=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    perturbation = torch.zeros(24393, 256).to(device)
    perturbation = torch.nn.Parameter(perturbation,requires_grad=True)
    optimizer = optim.Adam([perturbation], lr=learning_rate)
    old_centers = old_centers.to(device)
    lr = learning_rate
    for epoch in range(num_epochs):
        perturbed_centers = old_centers + perturbation
        perturbed_centers = perturbed_centers.to(device)
        data_loader = DataLoader(perturbed_centers, batch_size=batch_size, shuffle=True)

        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            indices = torch.randint(0, len(batch), (2, batch_size))
            ni, nj = batch[indices[0]], batch[indices[1]]
            dot_product = torch.sum(ni * nj, dim=1)
            loss = torch.mean(torch.max(dot_product-threshold, torch.tensor(0.0))).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            perturbed_centers = old_centers + perturbation
            updated_data_loader = DataLoader(perturbed_centers, batch_size=batch_size, shuffle=True)
            # 在下一次迭代之前更新 data_loader ?
            data_loader = updated_data_loader

            if (batch_idx + 1) % print_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}],lr:{lr} Loss: {loss.item()}')
        if (epoch + 1) % 300 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                lr = param_group['lr']
    return perturbation

# disturbed_centers = train_loop(perturbation,old_centers,1024,100,0.001,0.5)

def odpp_old(cfg,perturbation,optimizer,old_centers, batch_size, num_epochs, learning_rate, threshold,device,print_interval=1):
    # targets = torch.rand(24393, 256).to(device)
    # centers = copy.deepcopy(old_centers)
    lr = learning_rate
    for epoch in range(num_epochs):
        perturbed_centers =  perturbation(old_centers)
        # perturbed_centers = perturbed_centers.to(device)
        sampler = DistributedSampler(perturbed_centers) if cfg.NUM_GPUS > 1 else RandomSampler(perturbed_centers)
        data_loader = DataLoader(perturbed_centers, batch_size=batch_size,sampler=sampler,drop_last=True)
        for batch_idx, batch in enumerate(data_loader):
            batch = batch
            indices = torch.randint(0, len(batch), (2, batch_size))
            ni, nj = batch[indices[0]], batch[indices[1]]
            dot_product = torch.sum(ni * nj, dim=1)
            loss = torch.mean(torch.max(dot_product-threshold, torch.tensor(0.0)))
            # print(loss.device)
            # loss = torch.sum(perturbation(targets))
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            # print(torch.all(perturbation.module.perturbation[0]==0))
            optimizer.step()
            perturbed_centers =  perturbation(old_centers)
            # updated_data_loader = DataLoader(perturbed_centers, batch_size=batch_size, shuffle=True)
            # # 在下一次迭代之前更新 data_loader ?
            # data_loader = updated_data_loader

            if (batch_idx + 1) % print_interval == 0 and torch.cuda.current_device() == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}],lr:{lr} Loss: {loss.item()}')
        if (epoch + 1) % 100 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                lr = param_group['lr']
    disturbed_centers =  perturbation(old_centers)
    return disturbed_centers

class Perturbation(nn.Module):
    def __init__(self):
        super(Perturbation, self).__init__()
        self.perturbation = nn.ParameterList([torch.nn.Parameter(torch.zeros(24393, 256),requires_grad=True)])
        # self.register_parameter('perturbation',self.perturbation)

    def forward(self,x):
        return self.perturbation[0] + x
    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # perturbation = torch.zeros(24393, 256).to(device)
# # perturbation = torch.nn.Parameter(perturbation,requires_grad=True)
# perturbation = Perturbation().to(device)
# old_centers = load_old_centers().to(device)
# optimizer = optim.Adam(perturbation.parameters(), lr=0.01)
# disturbed_centers = odpp_old(perturbation,optimizer,old_centers,1024,10,0.01,0.5,device)


# Train the perturbation
# perturbation = train_loop(perturbation, centers, print_interval=5)

# # Apply the perturbation to the centers
# centers = old_centers + perturbation
# # save_centers(centers)
# np.save(f'./old_centers_2/perturbed_old_centers_threshold_{threshold}_{batchsize}_epoch_{num_epoch}.npy',centers.detach().numpy())

# cosine_sim_matrix = calculate_cosine_similarity_matrix(centers.detach().numpy())
# num_similar_pairs = np.sum(cosine_sim_matrix > threshold)
# print(num_similar_pairs)


    
def odpp_new(cfg,perturbation,optimizer,old_centers,new_centers,batch_size, num_epochs, learning_rate, threshold,device,print_interval=1):
    lr = learning_rate
    sampler_new = DistributedSampler(new_centers) if cfg.NUM_GPUS > 1 else RandomSampler(new_centers)
    data_loader_new = DataLoader(new_centers, batch_size=batch_size,sampler=sampler_new,drop_last=True)
    for epoch in range(num_epochs):
        perturbed_centers = perturbation(old_centers)
        sampler = DistributedSampler(perturbed_centers) if cfg.NUM_GPUS > 1 else RandomSampler(perturbed_centers)
        data_loader = DataLoader(perturbed_centers, batch_size=batch_size,sampler=sampler,drop_last=True)
        loss = 0.0
        random.seed(epoch)
        indices = torch.randint(0, batch_size, (2,batch_size))
        for batch_idx, batch in enumerate(data_loader):
            # batch = batch.to(device)
            # ni = batch
            # indices_old = torch.randint(0, len(batch), batch_size)
            # ni = batch[indices_old]
            ni_old,nj_old = batch[indices[0]], batch[indices[1]]
            dot_product_old = torch.sum(ni_old * nj_old, dim=1)
            loss_old = torch.mean(torch.max(dot_product_old-threshold, torch.tensor(0.0)))
            loss_old.requires_grad_(True)
            for batch_idx_new, batch_new in enumerate(data_loader_new):
                # batch_new = batch_new.to(device)
                loss_new = 0.0
                nj_new = batch_new[indices[1]]
                # nj = batch_new
                dot_product_new = torch.sum(ni_old * nj_new, dim=1)
                loss_new += torch.mean(torch.max(dot_product_new-threshold, torch.tensor(0.0)))
                loss_new.requires_grad_(True)
            loss = loss + cfg.ODPP.SCALE_NEW * loss_new + cfg.ODPP.SCALE_OLD * loss_old
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            # print(torch.all(perturbation.module.perturbation[0].grad==0))
            optimizer.step()
            perturbed_centers = perturbation(old_centers)
            # updated_data_loader = DataLoader(perturbed_centers, batch_size=batch_size, shuffle=True,drop_last=True)
            # # 在下一次迭代之前更新 data_loader ?
            # data_loader = updated_data_loader

            if (batch_idx + 1) % print_interval == 0 and torch.cuda.current_device() == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}],lr:{lr} Loss: {loss.item()}')
        if (epoch + 1) % 300 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                lr = param_group['lr']
    disturbed_centers = perturbation(old_centers)
    return disturbed_centers        
