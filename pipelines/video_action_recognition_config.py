"""
Default setting in training/testing
Reference: https://github.com/dmlc/gluon-cv
"""
from yacs.config import CfgNode as CN


_C = CN()

# ---------------------------------------------------------------------------- #
# Distributed DataParallel setting: DDP_CONFIG
# ---------------------------------------------------------------------------- #

_C.DDP_CONFIG = CN(new_allowed=False)
# Number of nodes for distributed training
_C.DDP_CONFIG.WORLD_SIZE = 1
# Node rank for distributed training
_C.DDP_CONFIG.WORLD_RANK = 0
# Number of GPUs to use
_C.DDP_CONFIG.GPU_WORLD_SIZE = 8
# GPU rank for distributed training
_C.DDP_CONFIG.GPU_WORLD_RANK = 0
# Master node
_C.DDP_CONFIG.DIST_URL = 'tcp://127.0.0.1:10001'
# A list of IP addresses for each node
_C.DDP_CONFIG.WORLD_URLS = ['127.0.0.1']
# Whether to turn on automatic ranking match.
_C.DDP_CONFIG.AUTO_RANK_MATCH = True
# distributed backend
_C.DDP_CONFIG.DIST_BACKEND = 'nccl'
# Current GPU id.
_C.DDP_CONFIG.GPU = 0
# Whether to use distributed training or simply use dataparallel.
_C.DDP_CONFIG.DISTRIBUTED = True

# ---------------------------------------------------------------------------- #
# Standard training/testing setting: CONFIG
# ---------------------------------------------------------------------------- #

_C.CONFIG = CN(new_allowed=True)

_C.CONFIG.VAL = CN(new_allowed=True)
# Per GPU mini-batch size.
_C.CONFIG.VAL.BATCH_SIZE = 8


_C.CONFIG.INFERENCE = CN(new_allowed=True)
# Whether to extract features or make predictions.
# If set to True, only features will be returned.
_C.CONFIG.INFERENCE.FEAT = False

_C.CONFIG.DATA = CN(new_allowed=True)

# Paths of annotation files and actual data
_C.CONFIG.DATA.TRAIN_ANNO_PATH = ''
_C.CONFIG.DATA.TRAIN_DATA_PATH = ''
_C.CONFIG.DATA.VAL_ANNO_PATH = ''
_C.CONFIG.DATA.VAL_DATA_PATH = ''
# The number of classes to predict for the model.
_C.CONFIG.DATA.NUM_CLASSES = 80
# Whether to use multigrid training to speed up.
_C.CONFIG.DATA.MULTIGRID = False
# The number of frames of the input clip.
_C.CONFIG.DATA.CLIP_LEN = 32
# The video sampling rate of the input clip.
_C.CONFIG.DATA.FRAME_RATE = 2
# Whether to keep aspect ratio when resizing input
_C.CONFIG.DATA.KEEP_ASPECT_RATIO = False
# Temporal segment setting for training video action recognition models.
_C.CONFIG.DATA.NUM_SEGMENT = 1
_C.CONFIG.DATA.NUM_CROP = 1
# Multi-view evaluation for video action recognition models.
# Usually for 2D models, it is 25 segments with 10 crops.
# For 3D models, it is 10 segments with 3 crops.
# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.CONFIG.DATA.TEST_NUM_SEGMENT = 10
# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.CONFIG.DATA.TEST_NUM_CROP = 3
# The spatial crop size for training.
_C.CONFIG.DATA.CROP_SIZE = 224
# Size of the smallest side of the image during testing.
_C.CONFIG.DATA.SHORT_SIDE_SIZE = 256
# Pre-defined height for resizing input video frames.
_C.CONFIG.DATA.NEW_HEIGHT = 256
# Pre-defined width for resizing input video frames.
_C.CONFIG.DATA.NEW_WIDTH = 340
# Interpolation to resize image (random, bilinear, bicubic)
_C.CONFIG.DATA.INTERPOLATION = 'bicubic'


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.CONFIG.AUG = CN(new_allowed=True)
# Color jitter factor
_C.CONFIG.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.CONFIG.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.CONFIG.AUG.REPROB = 0.25
# Random erase mode
_C.CONFIG.AUG.REMODE = 'pixel'
# Random erase count
_C.CONFIG.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.CONFIG.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.CONFIG.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.CONFIG.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.CONFIG.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.CONFIG.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.CONFIG.AUG.MIXUP_MODE = 'batch'
# Label Smoothing
_C.CONFIG.AUG.LABEL_SMOOTHING = 0.1
# Repeated augmentation
_C.CONFIG.AUG.REPEATED_AUG = True

_C.CONFIG.AUG.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]
_C.CONFIG.AUG.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

_C.CONFIG.MODEL = CN(new_allowed=True)
# Model architecture. You can find available models in the model zoo.
_C.CONFIG.MODEL.NAME = ''
# Whether to load a checkpoint file. If True, please set the following PRETRAINED_PATH.
_C.CONFIG.MODEL.LOAD = False
# Whether to load FC from a checkpoint file.
_C.CONFIG.MODEL.LOAD_FC = True
# Path (a file path, or URL) to a checkpoint file to be loaded to the model.
_C.CONFIG.MODEL.PRETRAINED_PATH = ''
# Whether to use the trained weights in the model zoo.
_C.CONFIG.MODEL.PRETRAINED = False
# Whether to use pretrained backbone network. Usually this is set to True.
_C.CONFIG.MODEL.PRETRAINED_BASE = True
# BN options
_C.CONFIG.MODEL.BN_EVAL = False
_C.CONFIG.MODEL.PARTIAL_BN = False
_C.CONFIG.MODEL.BN_FROZEN = False
_C.CONFIG.MODEL.USE_AFFINE = False
# Dropout rate
_C.CONFIG.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.CONFIG.MODEL.DROP_PATH_RATE = 0.1
# Model backbone tune point
_C.CONFIG.MODEL.TUNE_POINT = 4
# Whether to output only keyfrmae
_C.CONFIG.MODEL.SINGLE_FRAME = False

# ViT backbone default configs
_C.CONFIG.ViT = CN(new_allowed=True)
_C.CONFIG.ViT.TUBELET_SIZE = 2
_C.CONFIG.ViT.PATCH_SIZE = 16
_C.CONFIG.ViT.IN_CHANS = 3
_C.CONFIG.ViT.EMBED_DIM = 768
_C.CONFIG.ViT.PRETRAIN_IMG_SIZE = 224
_C.CONFIG.ViT.USE_LEARNABLE_POS_EMB = False
_C.CONFIG.ViT.DROP_RATE = 0.
_C.CONFIG.ViT.ATTN_DROP_RATE = 0.
_C.CONFIG.ViT.DROP_PATH_RATE = 0.2  #
_C.CONFIG.ViT.DEPTH = 12
_C.CONFIG.ViT.NUM_HEADS = 12
_C.CONFIG.ViT.MLP_RATIO = 4
_C.CONFIG.ViT.QKV_BIAS = True
_C.CONFIG.ViT.QK_SCALE = None
_C.CONFIG.ViT.INIT_VALUES = 0.
_C.CONFIG.ViT.USE_CHECKPOINT = True
_C.CONFIG.ViT.LAYER_DECAY = 0.75
_C.CONFIG.ViT.WEIGHT_DECAY = 0.05
_C.CONFIG.ViT.NO_WEIGHT_DECAY = ['pos_embed']

_C.CONFIG.LOG = CN(new_allowed=True)
# Base directory where all output files are written
_C.CONFIG.LOG.BASE_PATH = ''
# Pre-defined name for each experiment.
# If set to 'use_time', the start time will be appended to the directory name.
_C.CONFIG.LOG.EXP_NAME = 'use_time'
# Directory where training logs are written
_C.CONFIG.LOG.LOG_DIR = 'tb_log'
# Directory where checkpoints are written
_C.CONFIG.LOG.SAVE_DIR = 'checkpoints'
# Directory where testing logs are written
_C.CONFIG.LOG.EVAL_DIR = ''
# Save a checkpoint after every this number of epochs
_C.CONFIG.LOG.SAVE_FREQ = 1
# Display the training log after every this number of iterations
_C.CONFIG.LOG.DISPLAY_FREQ = 1

_C.CONFIG.GRADIENT_CHECKPOINTING = False
_C.CONFIG.AMP = False

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for your project."""
    return _C.clone()
