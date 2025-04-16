#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode


# Global config object
_C = CfgNode()
# Example usage:
#   from configs.config import cfg

_C.USE_AMP = False
_C.DBG = False
_C.OUTPUT_DIR = "./output"
_C.RUN_N_TIMES = 5
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN_BENCHMARK = False

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.SEED = 42
_C.SYNC_BN = False
_C.BN_TRACK = True
_C.GATHER_ALL = True
_C.ONLY_EXTRACT_FEAT = False
# ----------------------------------------------------------------------
# Model options
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()
# 模型结构参数
_C.MODEL.TYPE = "vit"# resnet， convnext, vit, swin, ssl-vit (self-supervised learning)
_C.MODEL.PROJECTION_LAYERS = -1
_C.MODEL.PROJECTION_DIM = 256
_C.MODEL.TRANSFER_TYPE = "linear"  # one of end2end, adapter
_C.MODEL.MOMENTUM = False
_C.MODEL.MOMENTUM_M = 0.999
_C.MODEL.DROP = 0.0
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.PRETRAIN = False

_C.MODEL.DATA = CfgNode()# 与数据集相关结构参数
_C.MODEL.DATA.FEATURE = ""  # e.g. inat2021_supervised
_C.MODEL.DATA.TRAIN_RATIO = 1.0
_C.MODEL.DATA.CROPSIZE = 224  # or 384
_C.MODEL.DATA.NUMBER_CLASSES = -1

# pretrained/checkpoint相关参数
_C.MODEL.WEIGHT_PATH = ""  # if resume from some checkpoint file
_C.MODEL.MODEL_ROOT = ""  # root folder for pretrained model weights
_C.MODEL.SAVE_CKPT = False
_C.MODEL.OLD_PRETRAINED_PATH = ""

# ----------------------------------------------------------------------
# adapter options
# ----------------------------------------------------------------------
_C.MODEL.ADAPTER = CfgNode()
_C.MODEL.ADAPTER.REDUCATION_FACTOR = 8
_C.MODEL.ADAPTER.ATTN_REDUCATION_FACTOR = 32
_C.MODEL.ADAPTER.STYLE = "Pfeiffer"
_C.MODEL.ADAPTER.TYPE = "normal"# LoRA
_C.MODEL.ADAPTER.ADAPTER_LAYERNORM_OPTION = "in"# AdaptFormer
_C.MODEL.ADAPTER.ADAPTER_SCALAR = 0.1# AdaptFormer
_C.MODEL.ADAPTER.ATTN_ADAPTER_SCALAR = 0.01
_C.MODEL.ADAPTER.DROPOUT = 0.1# AdaptFormer
_C.MODEL.ADAPTER.INIT_OPTION = "lora"# AdaptFormer
_C.MODEL.ADAPTER.LoRA_R = 0
_C.MODEL.ADAPTER.LoRA_ALPHA = 1
_C.MODEL.ADAPTER.LoRA_DROPOUT = 0.
_C.MODEL.ADAPTER.LoRA_HEADS = 8

# ----------------------------------------------------------------------
# switchnet options
# ----------------------------------------------------------------------
_C.MODEL.BACKBONE = CfgNode()

_C.MODEL.BACKBONE.NAME = "build_switch_resnet_backbone"
_C.MODEL.BACKBONE.DEPTH = "18x"
_C.MODEL.BACKBONE.LAST_STRIDE = 1
# Backbone feature dimension
_C.MODEL.BACKBONE.FEAT_DIM = 512
# Normalization method for the convolution layers.
_C.MODEL.BACKBONE.NORM = "BN"
# If use IBN block in backbone
_C.MODEL.BACKBONE.WITH_IBN = False
# If use SE block in backbone
_C.MODEL.BACKBONE.WITH_SE = False
# If use Non-local block in backbone
_C.MODEL.BACKBONE.WITH_NL = False
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = True
# Pretrain model path
_C.MODEL.BACKBONE.PRETRAIN_PATH = ''

_C.SNET = CfgNode()
_C.SNET.WIDTH_MULT_LIST = [1.0]
_C.SNET.LAMBDA = 0.0005
_C.SNET.USE_SFSC_LOSS = True

# ----------------------------------------------------------------------
# New model options
# ----------------------------------------------------------------------


_C.UPDATE = CfgNode()
_C.UPDATE.HOT_REFRESH = False
_C.UPDATE.TRASLATOR_TYPE = ""
_C.UPDATE.HIDDEN_DIM = 768
_C.UPDATE.HIDDEN_LAYERS = 2
_C.UPDATE.LOSS = CfgNode()

# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.TYPE = "base"
_C.SOLVER.OPTIMIZER = "sgd"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.PATIENCE = 300 # 如果loss连续300个epoch没有下降就认为模型已经开始过拟合,停止训练

_C.SOLVER.SCHEDULER = "cosine"  # 使用余弦下降的方式衰减学习率。

_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_MULTIPLIER = 1. # for prompt + bias
_C.SOLVER.BACKBONE_MULTIPLIER = 1.
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.MILESTONES = [30, 60, 120]
_C.SOLVER.WARMUP_STEP = 200  
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000
_C.SOLVER.DBG_TRAINABLE = False # if True, will print the name of trainable params
_C.SOLVER.WARMUP_EPOCH = 0 

_C.SOLVER.THRESHOLD = 0.5 # clone from advBCT


_C.LOSS = CfgNode()
_C.LOSS.TYPE = 'softmax'# arcface, cosface, contrastive
_C.LOSS.USE_XBM = False
_C.LOSS.SCALE = 30.0
_C.LOSS.MARGIN = 0.3
_C.LOSS.ONLY_COMPATIBLE = False
_C.LOSS.LOSS_POS_MARGIN = 1.0
_C.LOSS.LOSS_NEG_MARGIN = 0.5

_C.COMP_LOSS = CfgNode()
_C.COMP_LOSS.TYPE = 'bct'
_C.COMP_LOSS.USE_XBM = False
_C.COMP_LOSS.NEW_CLASS_PROTOTYPE = "using_old" # "null" / "using_new"
_C.COMP_LOSS.NORM_PROTOTYPE = True
_C.COMP_LOSS.ALL_CLASS_CENTERS = False
_C.COMP_LOSS.TOPK_FOR_SUPPORT = 100
_C.COMP_LOSS.ELASTIC_SCALE = 0.4
_C.COMP_LOSS.QUEUE_CAPACITY = 12
_C.COMP_LOSS.ELASTIC_BOUNDARY = False
_C.COMP_LOSS.TEMPERATURE = 0.01
_C.COMP_LOSS.TRIPLET_MARGIN = 0.8
_C.COMP_LOSS.TOPK_NEG = 10
_C.COMP_LOSS.WEIGHT = 1.0
_C.COMP_LOSS.DISTILLATION_TEMP = 0.01
_C.COMP_LOSS.FOCAL_BETA = 5.0
_C.COMP_LOSS.FOCAL_ALPHA = 1.0
_C.COMP_LOSS.NEW_SCALE = 0.01
_C.COMP_LOSS.PERTURBATION_BOUNDARY = False
_C.COMP_LOSS.PERTURBATION_BOUNDARY_THRESHOLD = 0.9
_C.COMP_LOSS.LEARNING_PERTURBATION = False
_C.COMP_LOSS.LEARNING_PERTURBATION_PATH = ''
_C.COMP_LOSS.ONLY_TEST = False 

_C.ODPP = CfgNode()
_C.ODPP.USE_ODPP = False     # use optimizor driven prototypes purturbation
_C.ODPP.BATCHSIZE = 1024
_C.ODPP.EPOCH_NUM = 100
_C.ODPP.LR = 0.001
_C.ODPP.THRESHOLD_OLD = 0.9
_C.ODPP.THRESHOLD_NEW = 0.9
_C.ODPP.SCALE_OLD = 1
_C.ODPP.SCALE_NEW = 1

_C.UPGRADE_LOSS = CfgNode()
_C.UPGRADE_LOSS.TYPE = 'base'
_C.UPGRADE_LOSS.WEIGHT = 1

_C.SUB_MODEL = CfgNode()
_C.SUB_MODEL.SPARSITY = [0.2,0.4,0.6,0.8]
_C.SUB_MODEL.COM_LOSS_SCALE = 1.0
_C.SUB_MODEL.ONLY_TEST_PARENT = False
_C.SUB_MODEL.DEEPCOPY = False
_C.SUB_MODEL.BCT_S = False
_C.SUB_MODEL.GRAD_SCALE = False
_C.SUB_MODEL.SLICE_SUBMODEL = False
_C.SUB_MODEL.SUB_MODEL_LOSS_TYPE = 'cls'
_C.SUB_MODEL.RANDOM_SPARSITY = False
_C.SUB_MODEL.FIXBN = False
_C.SUB_MODEL.FREEZEW_M = False
_C.SUB_MODEL.GRAD_PROJ = False
_C.SUB_MODEL.ADD_KL_LOSS = False
_C.SUB_MODEL.ADD_KL_SCALE = False
_C.SUB_MODEL.FIX_SUBMODEL = False
_C.SUB_MODEL.PROJ_W_M = False
_C.SUB_MODEL.PROJ_G_J = False
_C.SUB_MODEL.ADD_KL_NOISE = False
_C.SUB_MODEL.ADD_LOSS_SCALE = False
_C.SUB_MODEL.PROJ_BY_KERNEL = True
_C.SUB_MODEL.ONLY_PROJ_W_M = False
_C.SUB_MODEL.CHANGE_KL_SCALE = False
_C.SUB_MODEL.PARETO_AUG = False
_C.SUB_MODEL.ONLY_PARETO = False
_C.SUB_MODEL.ADD_HARD_SCALE = False
_C.SUB_MODEL.PARENT_MODEL_WEIGHT = ''
_C.SUB_MODEL.SCALE_START_EPCOH = 30
_C.SUB_MODEL.ADD_PARENT_HARD_SCALE = False
_C.SUB_MODEL.PARENT_SPARSITY = 0.0
_C.SUB_MODEL.ADAPTIVE_BN = False
_C.SUB_MODEL.ADD_PARETO_SCALE = False
_C.SUB_MODEL.ADD_PARETO_GAMMA = False
_C.SUB_MODEL.USE_SWITCHNET = False
_C.SUB_MODEL.MUL_SCORE_MAP = False
_C.SUB_MODEL.COS_SIM = CfgNode()
_C.SUB_MODEL.COS_SIM.EXP = 1.0
_C.SUB_MODEL.COS_SIM.USE_DOT = False

# ----------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.DATASET_TYPE = "landmark"
_C.DATA.NAME = ""
_C.DATA.SPLIT = ""
_C.DATA.DATAPATH = ""

#gldv2
_C.DATA.FILE_DIR = ''

_C.DATA.PERCENTAGE = 1.0
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"
_C.DATA.TRAIN_IMG_LIST = None

_C.DATA.RANDOM = True # 
_C.DATA.MPerClass = None  # borrow from image retrieval transformer
_C.DATA.LENGTH = None

_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA.PIN_MEMORY = True
_C.DATA.MEMORY_RATIO = 1.0


# 与模型结构相关的数据集参数，已经放到MODEL.DATA下
#_C.DATA.FEATURE = ""  # e.g. inat2021_supervised
#_C.DATA.NUMBER_CLASSES = -1
#_C.DATA.TRAIN_RATIO = 1.0
#_C.DATA.CROPSIZE = 224  # or 384

_C.EVAL = CfgNode()
_C.EVAL.RECALL_RANK = [1, ]
_C.EVAL.INTERVAL = 1
_C.EVAL.DISTANCE_NAME = 'cosine' #or "l2"
_C.EVAL.WEIGHT_PATH = ''
_C.EVAL.SKIP_EVAL = False

#gldv2
_C.EVAL.DATASET = ''
_C.EVAL.ROOT = ''
_C.EVAL.SAVE_DIR = ''
_C.EVAL.OLD_SAVE_FILE = ''
_C.EVAL.HOT_REFRESH = False
_C.EVAL.OLD_SAVE_FILE = ""
_C.EVAL.PRETRAINED_PATH = ""

# from fastreid
_C.INPUT = CfgNode()
# Size of the image during training
# _C.INPUT.SIZE_TRAIN = [256, 128]
_C.INPUT.SIZE_TRAIN = [224, 224]
# Size of the image during test
# _C.INPUT.SIZE_TEST = [256, 128]
_C.INPUT.SIZE_TEST = [224, 224]

# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5

# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10

# Random color jitter
_C.INPUT.CJ = CfgNode()
_C.INPUT.CJ.ENABLED = False
_C.INPUT.CJ.PROB = 0.5
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1

# Random Affine
_C.INPUT.DO_AFFINE = False

# Auto augmentation
_C.INPUT.DO_AUTOAUG = False
_C.INPUT.AUTOAUG_PROB = 0.0

# Augmix augmentation
_C.INPUT.DO_AUGMIX = False
_C.INPUT.AUGMIX_PROB = 0.0

# Random Erasing
_C.INPUT.REA = CfgNode()
_C.INPUT.REA.ENABLED = False
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.VALUE = [0.485*255, 0.456*255, 0.406*255]
# Random Patch
_C.INPUT.RPT = CfgNode()
_C.INPUT.RPT.ENABLED = False
_C.INPUT.RPT.PROB = 0.5

_C.TEST = CfgNode()

_C.TEST.EVAL_PERIOD = 20

# Number of images per batch in one process.
_C.TEST.IMS_PER_BATCH = 64
_C.TEST.METRIC = "cosine"
_C.TEST.ROC_ENABLED = False
_C.TEST.FLIP_ENABLED = False

# Average query expansion
_C.TEST.AQE = CfgNode()
_C.TEST.AQE.ENABLED = False
_C.TEST.AQE.ALPHA = 3.0
_C.TEST.AQE.QE_TIME = 1
_C.TEST.AQE.QE_K = 5

# Re-rank
_C.TEST.RERANK = CfgNode()
_C.TEST.RERANK.ENABLED = False
_C.TEST.RERANK.K1 = 20
_C.TEST.RERANK.K2 = 6
_C.TEST.RERANK.LAMBDA = 0.3

# Precise batchnorm
_C.TEST.PRECISE_BN = CfgNode()
_C.TEST.PRECISE_BN.ENABLED = False
_C.TEST.PRECISE_BN.DATASET = 'Market1501'
_C.TEST.PRECISE_BN.NUM_ITER = 300

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CfgNode()
# P/K Sampler for data loading
_C.DATALOADER.PK_SAMPLER = True
# Naive sampler which don't consider balanced identity sampling
_C.DATALOADER.NAIVE_WAY = True
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8
def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
