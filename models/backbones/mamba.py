import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict

from timm.models import create_model
import argparse

from VideoMamba.videomamba.video_sm.models.videomamba import (
    videomamba_tiny, 
    videomamba_small, 
    videomamba_middle, 
)
import VideoMamba.videomamba.video_sm.models.videomamba as videomamba_models
def build_mamba(config):
    delete_head = True
    num_frames = config.get("CONFIG", {}).get("DATA", {}).get("TEMP_LEN", 16)
    tubelet_size = 1
    input_size = config.get("CONFIG", {}).get("DATA", {}).get("IMG_SIZE", 224)
    nb_classes = config.get("CONFIG", {}).get("DATA", {}).get("NUM_CLASSES", 80)
    orig_t_size = num_frames

    mamba_version = config.get("CONFIG", {}).get("MODEL", {}).get('BACKBONE_TYPE', 'videomamba_tiny')
    ckpt_path = config.get("CONFIG", {}).get("MAMBA", {}).get('PRETRAIN', {}).get(mamba_version, None)
    use_checkpoint = False
    checkpoint_num = 0

    print(f"[INFO] Mamba version: {mamba_version}")
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    model = getattr(videomamba_models, mamba_version)(
        img_size=input_size,
        pretrained=False,
        num_classes=nb_classes,
        fc_drop_rate=0.0,
        drop_path_rate=0.4,
        kernel_size=1,
        num_frames=num_frames,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
    )
    
    patch_size = model.patch_embed.patch_size
    window_size = (num_frames // tubelet_size, input_size // patch_size[0], input_size // patch_size[1])

    checkpoint_model = None
    for model_key in 'model|module'.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    if 'head.weight' in checkpoint_model:
        if delete_head:
            print("Removing head from pretrained checkpoint")
            del checkpoint_model['head.weight']
            del checkpoint_model['head.bias']
        elif checkpoint_model['head.weight'].shape[0] == 710:
            if nb_classes == 400:
                checkpoint_model['head.weight'] = checkpoint_model['head.weight'][:nb_classes]
                checkpoint_model['head.bias'] = checkpoint_model['head.bias'][:nb_classes]
            elif nb_classes in [600, 700]:
                map_path = f'k710/label_mixto{nb_classes}.json'
                print(f'Load label map from {map_path}')
                with open(map_path) as f:
                    label_map = json.load(f)
                checkpoint_model['head.weight'] = checkpoint_model['head.weight'][label_map]
                checkpoint_model['head.bias'] = checkpoint_model['head.bias'][label_map]

    new_dict = OrderedDict()
    for key in checkpoint_model.keys():
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict


    # interpolate position embedding
    
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
    num_patches = model.patch_embed.num_patches # 
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # B, L, C -> B, H, W, C -> B, C, H, W
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # B, C, H, W -> B, H, W, C ->  B, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_size, new_size, embedding_size) 
        pos_tokens = pos_tokens.flatten(1, 2) # B, L, C
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
    
    # we use 8 frames for pretraining
    temporal_pos_embed = checkpoint_model['temporal_pos_embedding']
    orig_t_size = orig_t_size // model.patch_embed.tubelet_size
    new_t_size = num_frames // model.patch_embed.tubelet_size
    # height (== width) for the checkpoint position embedding
    if orig_t_size != new_t_size:
        print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
        temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
        temporal_pos_embed = torch.nn.functional.interpolate(
            temporal_pos_embed, size=(new_t_size,), mode='linear', align_corners=False
        )
        temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
        checkpoint_model['temporal_pos_embedding'] = temporal_pos_embed
    load_state_dict(model, checkpoint_model, prefix='')

    model.out_channels = config.get("CONFIG", {}).get("MAMBA", {}).get('OUT_CHANNELS', {}).get(mamba_version, 192)
    return model



def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))