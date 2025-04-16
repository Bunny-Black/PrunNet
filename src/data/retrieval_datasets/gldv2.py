import os
import os.path as osp
from .base_dataset import BaseDataset
from typing import Sequence
from PIL import Image
import pickle
#import jpeg4py, 
import torch
import numpy as np
import cv2
cv2.setNumThreads(0)
import src.data.lmdb_utils as lmdb_utils
import torchvision.transforms as transforms

class gldv2Dataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super(gldv2Dataset, self).__init__(*args, **kwargs)
        assert self.split in {"train", "query", "gallery"}

    def set_paths_and_labels(self):
        root = self.cfg.DATA.DATAPATH
        flist = self.cfg.DATA.FILE_DIR
        if type(flist) is str:
            self.imlist = files_reader(flist,root)
        elif isinstance(flist,Sequence):
            self.imlist = flist
        else:
            raise BaseException('Gldv2TrainDataset: flist must be a txt file or list!')
        self.paths, self.labels = [], []
        for (path, label) in self.imlist:
            self.paths.append(path)
            self.labels.append(label)

    def __getitem__(self, index):
        path, label = self.imlist[index]
        img = self._reader(path)
        if self.transform:
            img = self.transform(img)
        return img,label,index
    
    def __len__(self):
        return len(self.imlist)

    def _reader(self,path):
        return Image.open(path)

    
class gldv2Dataset_lmdb(BaseDataset):
    
    def __init__(self, cfg, data_dir, input_size=224, split="train"):
        super(gldv2Dataset_lmdb, self).__init__(cfg, data_dir, input_size=input_size, split=split)
        assert self.split in {"train", "query", "gallery"}
        is_training = split == "train" and (not cfg.ONLY_EXTRACT_FEAT)
        self.transform = build_gldv2_transform_lmdb(input_size, is_training)
        self.cfg = cfg
        if self.cfg.COMP_LOSS.TYPE == 'SSPL':
            from ...utils.sspl.Anchor_points_generation import load_pickle
            self.teacher_feat = np.ascontiguousarray(load_pickle('/data/sunyushuai/FCL_submodel/FCL_ViT/train_features/GLDv2_30%.pkl')['train'], dtype='float32')
        
    def set_paths_and_labels(self):
        root = self.cfg.DATA.DATAPATH
        flist = self.cfg.DATA.FILE_DIR
        if type(flist) is str:
            self.imlist = files_reader_lmdb(flist, root)
        elif isinstance(flist,Sequence):
            self.imlist = flist
        else:
            raise BaseException('Gldv2TrainDataset: flist must be a txt file or list!')
        self.paths, self.labels = [], []
        for (path, label) in self.imlist:
            self.paths.append(path)
            self.labels.append(label)

    def __getitem__(self, index):
        path, label = self.imlist[index]
        img = lmdb_utils.decode_img(self.data_dir, path)
        if self.transform:
            img = self.transform(img)
        if self.cfg.COMP_LOSS.TYPE == 'SSPL':
            teacher_feature = self.teacher_feat[index]
            return img,label,index,teacher_feature
        else:
            return img, label, index 
            
def files_reader(path, root):
    flist = []
    with open(path) as f:
        for line in f.readlines():
            [imid,label]=line.split()
            flist.append([osp.join(root,imid),int(label)])
    return flist

def files_reader_lmdb(path, root):
    flist = []
    with open(path) as f:
        for line in f.readlines():
            [imid,label]=line.split()
            flist.append(['/{:s}'.format(imid), int(label)])
    return flist

# def jpeg4py_loader(path):
#     """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
#     try:
#         return jpeg4py.JPEG(path).decode()# 读取的应该是RGB
#     except Exception as e:
#         print('ERROR: Could not read image "{}"'.format(path))
#         print(e)
#         return None

def build_gldv2_transform_lmdb(input_size=224, is_training=False):
    t = []
    t.append(transforms.ToPILImage())
    t.append(transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC))
    if is_training:    
        t.append(transforms.RandomCrop(input_size))
        t.append(transforms.RandomHorizontalFlip(0.5))
    else:
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(t)


class ROxfordParisTestDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, root, query_flag=True, *args, **kwargs):
        super(ROxfordParisTestDataset, self).__init__(cfg = cfg, data_dir = root, *args, **kwargs)
        self.root = root
        dataset_name = dataset_name.lower()
        self.query_flag = query_flag
        gnd_fname = osp.join(self.root,'gnd_{}.pkl'.format(dataset_name))
        if dataset_name not in ['roxford5k','rparis6k','revisitop1m']:
            raise Exception('ROxfordParisTestDataset: only support roxford5k, rparis6k and revisitop1m')
        if dataset_name in ['roxford5k','rparis6k']:
            with open(gnd_fname,'rb') as f:
                config = pickle.load(f)
            config['gnd_fname'] = gnd_fname
            config['ext'] = '.jpg'
            config['qext'] = '.jpg'
        else:
            config = {}
            config['imlist_fname'] = osp.join(self.root, '{}.txt'.format(dataset_name))
            config['imlist'] = self.read_imlist(config['imlist_fname'])
            config['qimlist'] = []
            config['ext'] = ''
            config['qext'] = ''

        config['dir_data'] = osp.join(root, dataset_name)
        config['dir_images'] = osp.join(config['dir_data'], 'jpg')

        config['n'] = len(config['imlist'])
        config['nq'] = len(config['qimlist'])
        self.config = config
        if query_flag:
            self.imlist = self.config['qimlist']
        else:
            self.imlist = self.config['imlist']
        self.query_gts = ''
        if query_flag:
            self.query_gts = config['gnd']

    def read_imlist(self,imlist_fn):
        with open(imlist_fn, 'r') as file:
            imlist = file.read().splitlines()
        return imlist

    def config_imname(self,i):
        return osp.join(self.config['dir_images'], self.imlist[i] + self.config['qext'] if self.query_flag else self.imlist[i] + self.config['ext'])

    def __getitem__(self, index):
        path = self.config_imname(index)
        img = self._reader(path)
        img = self.transform(img)
        return img,index,index

    def _reader(self,path):
        return Image.open(path)
    
    def __len__(self):
        if self.query_flag:
            return len(self.config['qimlist'])
        else:
            return len(self.config['imlist'])

class Gldv2TestDataset(BaseDataset):
    def __init__(self,cfg, dataset_name, root, query_flag=True, *args, **kwargs):
        super(Gldv2TestDataset, self).__init__(cfg, data_dir=root, *args, **kwargs)
        if query_flag:
            self.file = osp.join(root,"gldv2_private_query_list.txt")
        else:
            self.file = osp.join(root, "gldv2_gallery_list.txt")
        self.imlist = files_reader(self.file,root)
        self.query_gts_list = osp.join(root, "gldv2_private_query_gt.txt")
        self.query_flag = query_flag
        self.query_gts = [[], [], []]
        if query_flag:
            # [img_name: str, img_index: int, gts: int list]
            with open(self.query_gts_list, 'r') as f:
                for line in f.readlines():
                    img_name, img_index, tmp_gts = line.split(" ")
                    gts = [int(i) for i in tmp_gts.split(",")]
                    self.query_gts[0].append(img_name)
                    self.query_gts[1].append(int(img_index))
                    self.query_gts[2].append(gts)

    def __getitem__(self, index):
        path, label = self.imlist[index]
        img = self._reader(path)
        img = self.transform(img)
        return img,label,index

    def __len__(self):
        return len(self.imlist)

    def _reader(self,path):
        return Image.open(path)