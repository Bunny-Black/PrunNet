import os
import numpy as np
from torchvision import datasets
from .base_dataset import BaseDataset
import json

class Cub200Dataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super(Cub200Dataset, self).__init__(*args, **kwargs)
        assert self.split in {"train", "test"}

    def set_paths_and_labels(self):

        dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'images'))
        paths = np.array([a for (a, b) in dataset.imgs])
        labels = np.array([b for (a, b) in dataset.imgs])
        sorted_lb = list(sorted(set(labels)))
        if self.split == "train":
            set_labels = set(sorted_lb[:len(sorted_lb) // 2])
        else:
            set_labels = set(sorted_lb[len(sorted_lb) // 2:])
        self.paths = []
        self.labels = []
        for lb, pth in zip(labels, paths):
            if lb in set_labels:
                self.paths.append(pth)
                self.labels.append(lb)


class Cub200Dataset_Official_split(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(Cub200Dataset_Official_split, self).__init__(*args, **kwargs)
        assert self.split in {"train", "test",'gallery'}

    def set_paths_and_labels(self):
        if self.split == "train":
            # with open(os.path.join(self.data_dir,"train.json")) as f:
            with open(self.cfg.DATA.FILE_DIR) as f:
                data = json.load(f)
            paths = []
            labels = []
        
            for key,value in data.items():
                paths.append(os.path.join(self.data_dir,key))
                labels.append(int(value) - 1)   #label从0开始
            self.paths = paths
            self.labels = labels
        else:
            file = self.split + '.json'
            with open(os.path.join(self.cfg.EVAL.ROOT,file)) as f:
                data = json.load(f)
            paths = []
            labels = []
        
            for key,value in data.items():
                paths.append(os.path.join(self.data_dir,key))
                labels.append(int(value) - 1) #label从0开始
            self.paths = paths
            self.labels = labels