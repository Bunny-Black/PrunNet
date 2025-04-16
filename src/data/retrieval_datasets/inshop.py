import os

from .base_dataset import BaseDataset
from PIL import Image
import numpy as np


class InShopDataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super(InShopDataset, self).__init__(*args, **kwargs)
        assert self.split in {"train", "query", "gallery"}
        if self.cfg.COMP_LOSS.TYPE == 'SSPL':
            from ...utils.sspl.Anchor_points_generation import load_pickle
            self.teacher_feat = np.ascontiguousarray(load_pickle('/data/sunyushuai/FCL_submodel/FCL_ViT/train_features/inshop.pkl')['train'], dtype='float32')
            
    def set_paths_and_labels(self):
        if self.cfg.SOLVER.TYPE  == 'base':
            split_filename = "ordered_list_eval_partition_{:.2f}.txt".format(self.cfg.MODEL.DATA.TRAIN_RATIO)
        elif self.cfg.SOLVER.TYPE in ['compatible', 'adv_compatible','submodel_compatible']:
            split_filename = "ordered_list_eval_partition_{:.2f}.txt".format(self.cfg.NEW_MODEL.DATA.TRAIN_RATIO)
        else:
            raise NotImplementedError
        with open(os.path.join(self.data_dir, split_filename)) as f:
            db = f.read().split("\n")[2:-1]

        paths = []
        labels = []
        for line in db:
            line = line.split(" ")
            line = list(filter(lambda x: x, line))
            if line[2] == self.split:
                paths.append(os.path.join(self.data_dir, line[0]))
                labels.append(int(line[1].split("_")[-1]))

        self.paths = paths
        self.labels = labels
        
    def __getitem__(self, idx):
        pth = self.paths[idx]
        img = Image.open(pth).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        if self.cfg.COMP_LOSS.TYPE == 'SSPL' and self.split == 'train':
            teacher_feature = self.teacher_feat[idx]
            return img,label,idx,teacher_feature
        return img, label, idx