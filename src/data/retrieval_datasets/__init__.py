from .cub200 import Cub200Dataset, Cub200Dataset_Official_split
from .sop import SOPDataset
from .inshop import InShopDataset
from .imageNet import ImageNetDataset, ImageNet_list_Dataset
from .place365 import place365Dataset, place365_list_Dataset
from .gldv2 import gldv2Dataset, ROxfordParisTestDataset , Gldv2TestDataset, gldv2Dataset_lmdb
_gldv2_fact = {
    "roxford5k":ROxfordParisTestDataset,
    "rparis6k":ROxfordParisTestDataset,
    "gldv2":Gldv2TestDataset
}

def get_dataset(cfg):

    """returns train, query and gallery dataset"""

    train, query, gallery, test = None, None, None, None
    
    if cfg.DATA.NAME == 'retrieval-cub200' and cfg.DATA.SPLIT != "official":
        train   = Cub200Dataset(cfg, cfg.DATA.DATAPATH, split="train")
        query   = Cub200Dataset(cfg, cfg.DATA.DATAPATH, split="test")
        
    if cfg.DATA.NAME == 'retrieval-cub200' and cfg.DATA.SPLIT == "official":
        train   = Cub200Dataset_Official_split(cfg, cfg.DATA.DATAPATH, split="train")
        query   = Cub200Dataset_Official_split(cfg, cfg.DATA.DATAPATH, split="test")
        gallery = Cub200Dataset_Official_split(cfg, cfg.DATA.DATAPATH, split="gallery")

    if cfg.DATA.NAME == 'retrieval-sop':
        train   = SOPDataset(cfg, cfg.DATA.DATAPATH, split="train")
        query   = SOPDataset(cfg, cfg.DATA.DATAPATH, split="query")
        gallery   = SOPDataset(cfg, cfg.DATA.DATAPATH, split="gallery")

    if cfg.DATA.NAME == 'retrieval-inshop':
        train   = InShopDataset(cfg, cfg.DATA.DATAPATH, split="train")
        query   = InShopDataset(cfg, cfg.DATA.DATAPATH, split="query")
        gallery = InShopDataset(cfg, cfg.DATA.DATAPATH, split="gallery")

    if cfg.DATA.NAME == "retrieval-imageNet" and cfg.DATA.TRAIN_IMG_LIST is None:
        train = ImageNetDataset(cfg, cfg.DATA.DATAPATH, split="train")
        query = ImageNetDataset(cfg, cfg.DATA.DATAPATH, split="val")

    if cfg.DATA.NAME == "retrieval-imageNet" and cfg.DATA.TRAIN_IMG_LIST is not None:
        train = ImageNet_list_Dataset(cfg, cfg.DATA.DATAPATH, split="train")
        query = ImageNetDataset(cfg, cfg.DATA.DATAPATH, split="val")

    if cfg.DATA.NAME == "retrieval-place365" and cfg.DATA.TRAIN_IMG_LIST is None:
        train = place365Dataset(cfg, cfg.DATA.DATAPATH,split="train")
        query = place365Dataset(cfg, cfg.DATA.DATAPATH, split="val")
        
    if cfg.DATA.NAME == "retrieval-place365" and cfg.DATA.TRAIN_IMG_LIST is not None:
        train = place365_list_Dataset(cfg, cfg.DATA.DATAPATH, split="train")
        query = place365Dataset(cfg, cfg.DATA.DATAPATH, split="val")

    if cfg.DATA.NAME == "retrieval-gldv2" and cfg.DATA.FILE_DIR is not None:
        train   = gldv2Dataset(cfg, cfg.DATA.DATAPATH, split="train")
        query = _gldv2_fact[cfg.EVAL.DATASET](cfg, dataset_name = cfg.EVAL.DATASET, root = cfg.EVAL.ROOT, query_flag = True, split="query")
        gallery = _gldv2_fact[cfg.EVAL.DATASET](cfg, dataset_name = cfg.EVAL.DATASET, root = cfg.EVAL.ROOT, query_flag = False, split="gallery")
    
    if cfg.DATA.NAME == "lmdb-gldv2" and cfg.DATA.FILE_DIR is not None:
        train   = gldv2Dataset_lmdb(cfg, cfg.DATA.DATAPATH, split="train")
        query = _gldv2_fact[cfg.EVAL.DATASET](cfg, dataset_name = cfg.EVAL.DATASET, root = cfg.EVAL.ROOT, query_flag = True, split="query")
        gallery = _gldv2_fact[cfg.EVAL.DATASET](cfg, dataset_name = cfg.EVAL.DATASET, root = cfg.EVAL.ROOT, query_flag = False, split="gallery")
    
    return train, query, gallery, test

"""
def get_test_dataset(cfg):
    query, gallery = None, None

    if cfg.EVAL.DATASET in "rparis6k":
        query = _gldv2_fact[cfg.EVAL.DATASET](cfg, dataset_name = cfg.EVAL.DATASET, root = cfg.EVAL.ROOT, query_flag = True, split="query")
        gallery = _gldv2_fact[cfg.EVAL.DATASET](cfg, dataset_name = cfg.EVAL.DATASET, root = cfg.EVAL.ROOT, query_flag = False, split="gallery")
"""