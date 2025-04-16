from abc import abstractmethod
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import Counter

class BaseDataset(Dataset):

    def __init__(self, cfg, data_dir, input_size=224, split="train"):
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.split = split
        is_training = split == "train" and (not cfg.ONLY_EXTRACT_FEAT)
        self.transform = build_transform(input_size, is_training)
        self.set_paths_and_labels()

    @abstractmethod
    def set_paths_and_labels(self):
        pass

    def __len__(self,):
        return len(self.paths)

    def __getitem__(self, idx):
        pth = self.paths[idx]
        img = Image.open(pth).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label, idx

    def __repr__(self,):
        return f"{self.__class__.__name__}(split={self.split}, len={len(self)})"
    
    def get_class_num(self):
        if self.cfg.SOLVER.TYPE == 'base':
            return int(self.cfg.MODEL.DATA.NUMBER_CLASSES * self.cfg.MODEL.DATA.TRAIN_RATIO)
        elif self.cfg.SOLVER.TYPE == ['compatible', 'adv_compatible']:
            return int(self.cfg.NEW_MODEL.DATA.NUMBER_CLASSES * self.cfg.NEW_MODEL.DATA.TRAIN_RATIO)
        else:
            raise NotImplementedError
        
    def get_class_weights(self, weight_type, cls_num=-1, new2old_map=None):
        """get a list of class weight, return a list float"""
        if self.split != "train":
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self.split)
            )

        if cls_num == -1:
            cls_num = self.get_class_num()      
        if weight_type == "none":
            return [1.0] * cls_num
        else:
            raise ValueError("We do not support other class weights mode for now!")
            # 旧模型涉及到新旧类别映射问题
        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()


def build_transform(input_size=224, is_training=False):
    t = []
    t.append(transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC))
    if is_training:
        t.append(transforms.RandomCrop(input_size))
        t.append(transforms.RandomHorizontalFlip(0.5))
    else:
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(t)


