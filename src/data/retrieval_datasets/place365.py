import os
import numpy as np
from torchvision import datasets
from .base_dataset import BaseDataset

class place365Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(place365Dataset, self).__init__(*args, **kwargs)
        assert self.split in {"train", "val"}
    def set_paths_and_labels(self):
        dataset = datasets.ImageFolder(os.path.join(self.data_dir, self.split))
        paths = np.array([a for (a, b) in dataset.imgs])
        labels = np.array([b for (a, b) in dataset.imgs])
        self.paths = []
        self.labels = []
        for lb, pth in zip(labels, paths):
           self.paths.append(pth)
           self.labels.append(lb)
           
class place365_list_Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(place365_list_Dataset, self).__init__(*args, **kwargs)
        assert self.split in {"train"}
    def set_paths_and_labels(self):
        self.paths = [self.data_dir+"/train/"+line.split(' ')[0] for line in open(os.path.join(self.data_dir,self.cfg.DATA.TRAIN_IMG_LIST))]
        self.labels = [int(line.split(' ')[1].strip('\n')) for line in open(os.path.join(self.data_dir,self.cfg.DATA.TRAIN_IMG_LIST))]
