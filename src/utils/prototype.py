import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image

import os, glob
import math


def del_tensor_0_cloumn(Cs,device):

    idx = torch.all(Cs[..., :] == 0, axis=1)

    index=[]
    for i in range(idx.shape[0]):
        if not idx[i].item():
            index.append(i)


    index=torch.tensor(index).to(device)

    Cs = torch.index_select(Cs, 0, index)
    return Cs

@torch.no_grad()
def extract_prototype(cfg,old_model,train_loader,device,embedding_dim):
      old_model.eval()
      num_class = cfg.MODEL.DATA.NUMBER_CLASSES
      old_prototype = torch.zeros(num_class,embedding_dim).to(device)
      class_num = torch.zeros(num_class,1).to(device)
      
      for (images, targets) in tqdm(train_loader, total=len(train_loader), desc="query"):
            images = images.to(device)
            old_embedding, embeddings_k, outputs = old_model(images, return_feature=True)
            targets = targets.to(device, non_blocking=True)   
            B,D = old_embedding.shape
            for i in range(B):
                   old_prototype[targets[i]] = torch.add(old_embedding[i],old_prototype[targets[i]])
                   class_num[targets[i]] = torch.add(class_num[targets[i]],1)
      
      
      old_prototype = del_tensor_0_cloumn(old_prototype,device)
      class_num = del_tensor_0_cloumn(class_num,device)   
      old_prototype =  old_prototype/class_num
      return old_prototype
        
       
def get_retrieval_input(data):
        return data
