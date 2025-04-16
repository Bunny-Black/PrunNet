import torch
from torch.utils.data import RandomSampler
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data.distributed import DistributedSampler
from .retrieval_datasets import get_dataset
from torch.utils.data import DataLoader
from ..utils import comm
import collections.abc as container_abcs

def _constract_retrieval_loader(cfg, batch_size, shuffle, drop_last_for_train):
    dataset_train, dataset_query, dataset_gallery, dataset_test = get_dataset(cfg)
    data_loader_test = None
    print("number of images: ", len(dataset_train))
    # RandomSampler handles shuffling automatically
    '''
    PerClass采样:在每个epoch中,每个类别都有同样的训练样本数量,大类和小类获得同样的训练权重
    DistributedSampler:从整个数据集中随机采样出固定数量的样本indices,然后根据当前GPU的rank,将所有indices 分配给多个GPU,而不关心这些样本来自具体的类别。
    DistributedSampler 默认进行shuffle, seed默认为0
    '''
    if cfg.DATA.RANDOM:
        assert(cfg.DATA.MPerClass is None), "If using the random sampler, please set MPerClass to `None`"
        sampler_train = DistributedSampler(dataset_train) if cfg.NUM_GPUS > 1 else RandomSampler(dataset_train)
    else:
        print('cfg.DATA.MPerClass: ', cfg.DATA.MPerClass)
        # 实际打印了两张卡每一个batch采样的样本的id，发现是不相同的
        # assert(cfg.NUM_GPUS==1 & (cfg.DATA.MPerClass is not None)), "We do not support distributed training when using MPerClassSampler"
        sampler_train = MPerClassSampler(dataset_train.labels, m=cfg.DATA.MPerClass, 
                                         batch_size=cfg.DATA.BATCH_SIZE, 
                                         length_before_new_iter=len(dataset_train)//cfg.NUM_GPUS)

    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler = sampler_train,
        batch_size = batch_size,
        num_workers = cfg.DATA.NUM_WORKERS,
        pin_memory = cfg.DATA.PIN_MEMORY,
        drop_last = drop_last_for_train)

    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size = batch_size,
        num_workers = cfg.DATA.NUM_WORKERS,
        pin_memory = cfg.DATA.PIN_MEMORY,
        drop_last = False,
        shuffle = shuffle)

    data_loader_gallery = None
    if dataset_gallery is not None:
        data_loader_gallery = torch.utils.data.DataLoader(
        #data_loader_gallery = DataLoaderX(
            dataset_gallery,
            batch_size=batch_size,
            num_workers = cfg.DATA.NUM_WORKERS,
            pin_memory = cfg.DATA.PIN_MEMORY,
            drop_last = False,
            shuffle = shuffle
        )
        
    return data_loader_train, data_loader_query, data_loader_gallery, data_loader_test

def construct_retrieval_loader(cfg):
    if cfg.NUM_GPUS > 1:
        drop_last_for_train = True      # drop掉最后不足一个batch的数据。
    else:
        drop_last_for_train = False
    return _constract_retrieval_loader(
        cfg=cfg,
        batch_size=cfg.DATA.BATCH_SIZE,
        shuffle=False,
        drop_last_for_train=drop_last_for_train,
    )

def collate_function(batch):
    # 加载原始的图像和标签数据
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([torch.tensor(item[1], dim=0) for item in batch])
    image_idxs = torch.stack([torch.tensor(item[2], dim=0) for item in batch])
    # 返回未经过 transform 的一个 batch 样本数据和标签数据
    return images, labels, image_idxs

string_classes = str
int_classes = int
