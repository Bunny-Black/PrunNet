import os

import torch
from transformers import BertTokenizer
from transformers.adapters import BertAdapterModel, AutoAdapterModel
from transformers.adapters import ViTAdapterModel
from transformers.adapters import MAMConfig, AdapterConfig

if __name__ == '__main__':
    model = ViTAdapterModel.from_pretrained('google/vit-base-patch16-224-in21k')
    mamconfig = MAMConfig()
    #model = ViTAdapterModel(mamconfig)
    print(model)
    model.add_adapter('mam_config', config=mamconfig)
    print(model)