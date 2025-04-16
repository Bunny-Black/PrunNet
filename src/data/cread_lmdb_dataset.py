""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import argparse
import json
import os
import os.path as osp

import cv2
import lmdb
import numpy as np
from tqdm import tqdm

import lmdb_utils

MAP_SIZE = 1000000000000


def is_valid_image(image):
    if image is None:
        return False
    else:
        H, W, C = image.shape
        return H * W * C

def files_reader(path,root):
    flist = []
    with open(path) as f:
        for line in f.readlines():
            [imid,label]=line.split()
            flist.append([osp.join(root,imid),int(label)])
    return flist

def gldv2_all_to_lmdb(dataset_root, flist, output_path):
    os.makedirs(output_path, exist_ok=True)
    lmdb_hander = lmdb.open(output_path, map_size=MAP_SIZE)
    f = open(osp.join(output_path, "keys.txt"), 'a')
    if type(flist) is str:
        imlist = files_reader(flist, dataset_root)
    else:
        raise BaseException('Gldv2TrainDataset: flist must be a txt file or list!')
    for idx, (path, label) in enumerate(tqdm(imlist)):
        lmb = lmdb_hander.begin(write=True)
        img = cv2.imread(path)
        if is_valid_image(img):
            img_data = cv2.imencode('.jpg', img)[1].tobytes()
            img_key = path.replace(dataset_root, "").encode('utf-8')
            f.write(path.replace(dataset_root, "")+'\n')
            print(img_key)
            lmb.put(key=img_key, value=img_data)
        else:
            print(path)
        lmb.commit()
    lmdb_hander.close()
    f.close()


def lasot_to_lmdb(dataset_root, output_path, split_path):
    os.makedirs(output_path, exist_ok=True)
    lmdb_hander = lmdb.open(output_path, map_size=MAP_SIZE)
    
    seqs_list = np.loadtxt(osp.join(osp.dirname(__file__), split_path),
                           dtype=np.str_)
    
    for seq in tqdm(seqs_list):
        lmb = lmdb_hander.begin(write=True)
        # cls = seq.split("-")[0]
        seq_path = os.path.join(dataset_root, seq)
        imgs = os.listdir(os.path.join(seq_path, 'img'))
        file_list = os.listdir(seq_path)
        for file in file_list:
            file_path = osp.join(seq_path, file)
            if osp.isfile(file_path):
                print(file_path)
                with open(osp.join(file_path), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    value = "".join(lines)
                key = file_path.replace(dataset_root, "").encode()
                # print(key)
                lmb.put(key, value.encode('utf-8'))
        
        for img in tqdm(imgs):
            img_path = osp.join(seq_path, 'img', img)
            img = cv2.imread(img_path)
            if is_valid_image(img):
                img_data = cv2.imencode('.jpg', img)[1].tobytes()
                img_key = img_path.replace(dataset_root, "").encode('utf-8')
                # print(img_key)
                lmb.put(key=img_key, value=img_data)
            else:
                print(img_path)
        lmb.commit()
    lmdb_hander.close()


UNZIP_OUTPUT_PATH = "/gpfs/work5/0/prjs0370/zhouli/LaSOT/unzip"
ZIP_PATH = "/gpfs/work5/0/prjs0370/zhouli/LaSOT/zip"
lmdb_path = "/gpfs/work5/0/prjs0370/zhouli/dataset/lasot_lmdb"


def lasot_to_lmdb_cls(dataset_root, class_name, output_path):
    os.makedirs(output_path, exist_ok=True)
    lmdb_hander = lmdb.open(output_path, map_size=MAP_SIZE)
    
    seqs_list = os.listdir(os.path.join(dataset_root, class_name))
    
    for seq in tqdm(seqs_list):
        lmb = lmdb_hander.begin(write=True)
        cls = class_name
        seq_path = os.path.join(dataset_root, cls, seq)
        imgs = os.listdir(os.path.join(seq_path, 'img'))
        file_list = os.listdir(seq_path)
        for file in file_list:
            file_path = osp.join(seq_path, file)
            if osp.isfile(file_path):
                with open(osp.join(file_path), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    value = "".join(lines)
                key = file_path.replace(dataset_root, "").encode()
                # print(key)
                lmb.put(key, value.encode('utf-8'))
        
        for img in tqdm(imgs):
            img_path = osp.join(seq_path, 'img', img)
            img = cv2.imread(img_path)
            if is_valid_image(img):
                img_data = cv2.imencode('.jpg', img)[1].tobytes()
                img_key = img_path.replace(dataset_root, "").encode('utf-8')
                print(img_key)
                # print(img_key)
                lmb.put(key=img_key, value=img_data)
            else:
                print(img_path)
        lmb.commit()
    lmdb_hander.close()


if __name__ == '__main__':
    
    
    gldv2_output = "/data/GLDv2_test"
    gldv2_root = "/data/GLDv2"
    flist = "/data/GLDv2/debug_lmdb.txt"

    gldv2_all_to_lmdb(dataset_root=gldv2_root, flist=flist, output_path=gldv2_output)
    
    """
    f = open(osp.join(gldv2_output, "keys.txt"), 'r')
    keys = f.readlines()
    print(keys)
    for i, key in enumerate(keys):
        print(key.strip())
        img = lmdb_utils.decode_img(gldv2_output, key.strip())
        print(type(img), img.shape)
    """

    
    """
    flist = "/data/GLDv2/gldv2_train_old_30percent_class.txt"
    f = open(flist, 'r')
    f_lmdb = open("/data/GLDv2/debug_lmdb.txt", 'a')
    i = 0
    for line in f.readlines():
        f_lmdb.write(line)
        i += 1
        if i > 1000:
            break
    f.close()
    f_lmdb.close()
    """