# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
The code refers to https://github.com/facebookresearch/detr
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process
from models.position_encoding import build_position_encoding

from models.backbones.ir_CSN_50 import build_CSN
from models.backbones.ir_CSN_152 import build_CSN as build_CSN_152
from models.backbones.vit import build_ViT
from models.backbones.mamba import build_mamba

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, t, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class Backbone(nn.Module):

    def __init__(self, train_backbone: bool, num_channels: int, position_embedding, return_interm_layers, cfg):
        super().__init__()

        if cfg.CONFIG.MODEL.BACKBONE_NAME== 'CSN-152':
            print("CSN-152 backbone")
            self.body = build_CSN_152(cfg)
        elif cfg.CONFIG.MODEL.BACKBONE_NAME== 'CSN-50':
            print("CSN-50 backbone")
            self.body = build_CSN(cfg)
        elif cfg.CONFIG.MODEL.BACKBONE_NAME== 'VideoMamba':
            self.body = build_mamba(cfg)
        else:
            print("ViT-B backbone")
            self.body = build_ViT(cfg)
        self.position_embedding = position_embedding

        for name, parameter in self.body.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        self.ds = cfg.CONFIG.MODEL.SINGLE_FRAME
        use_ViT = "ViT" in cfg.CONFIG.MODEL.BACKBONE_NAME
        use_Mamba = "VideoMamba" in cfg.CONFIG.MODEL.BACKBONE_NAME
        self.use_ViT = use_ViT
        self.use_Mamba = use_Mamba
        if self.use_Mamba:
            self.mamba_proj = nn.Sequential(
                nn.Linear( self.body.out_channels*2, self.body.out_channels),
                nn.LayerNorm(self.body.out_channels),
                nn.GELU(),
                nn.Linear(self.body.out_channels, cfg.CONFIG.MODEL.D_MODEL, bias=False),
                nn.LayerNorm(cfg.CONFIG.MODEL.D_MODEL),
                nn.Linear(cfg.CONFIG.MODEL.D_MODEL, cfg.CONFIG.MODEL.D_MODEL, bias=False)
            )
        if return_interm_layers:
            if not use_ViT and not use_Mamba:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
                self.strides = [8, 16, 32]
                self.num_channels = [512, 1024, 2048]
                self.in_features = None
            elif use_Mamba:
                out_channel = cfg.CONFIG.MODEL.D_MODEL
                in_channels = [self.body.out_channels*2]*4
                self.strides = [8, 16, 32]
                self.num_channels = in_channels
                self.lateral_convs = nn.ModuleList()

                for idx, scale in enumerate([4, 2, 1, 0.5]):
                    dim = in_channels[idx]
                    if scale == 4.0:
                        layers = [
                            nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                            LayerNorm(dim // 2),
                            nn.GELU(),
                            nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                        ]
                        out_dim = dim // 4
                    elif scale == 2.0:
                        layers = [nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                        out_dim = dim // 2
                    elif scale == 1.0:
                        layers = []
                        out_dim = dim
                    elif scale == 0.5:
                        layers = [nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                        out_dim = dim
                    else:
                        raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
                    layers.extend(
                        [
                            nn.Conv3d(
                                out_dim,
                                out_channel,
                                kernel_size=1,
                                bias=False,
                            ),
                            LayerNorm(out_channel),
                            nn.Conv3d(
                                out_channel,
                                out_channel,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                        ]
                    )
                    layers = nn.Sequential(*layers)

                    self.lateral_convs.append(layers)
            else:
                out_channel = cfg.CONFIG.MODEL.D_MODEL
                in_channels = [cfg.CONFIG.ViT.EMBED_DIM]*4
                self.strides = [8, 16, 32]
                self.num_channels = in_channels
                self.lateral_convs = nn.ModuleList()

                for idx, scale in enumerate([4, 2, 1, 0.5]):
                    dim = in_channels[idx]
                    if scale == 4.0:
                        layers = [
                            nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                            LayerNorm(dim // 2),
                            nn.GELU(),
                            nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                        ]
                        out_dim = dim // 4
                    elif scale == 2.0:
                        layers = [nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                        out_dim = dim // 2
                    elif scale == 1.0:
                        layers = []
                        out_dim = dim
                    elif scale == 0.5:
                        layers = [nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                        out_dim = dim
                    else:
                        raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
                    layers.extend(
                        [
                            nn.Conv3d(
                                out_dim,
                                out_channel,
                                kernel_size=1,
                                bias=False,
                            ),
                            LayerNorm(out_channel),
                            nn.Conv3d(
                                out_channel,
                                out_channel,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                        ]
                    )
                    layers = nn.Sequential(*layers)

                    self.lateral_convs.append(layers)                
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        if not use_ViT and not use_Mamba:
            self.body = IntermediateLayerGetter(self.body, return_layers=return_layers)
        self.backbone_name = cfg.CONFIG.MODEL.BACKBONE_NAME

    def space_forward(self, features):
        mapped_features = {}
        for i, feature in enumerate(features):
            if isinstance(feature, NestedTensor):
                mask = feature.mask
                feature = feature.tensors
                mapped_features.update({f"{i}": NestedTensor(self.lateral_convs[i](feature), mask)})
            else:
                mapped_features.update({f"{i}": self.lateral_convs[i](feature)})
            
        return mapped_features

    def forward(self, tensor_list: NestedTensor):
        if "SlowFast" in self.backbone_name:
            xs, xt = self.body([tensor_list.tensors[:, :, ::4, ...], tensor_list.tensors])
        elif "TPN" in self.backbone_name:
            xs, xt = self.body(tensor_list.tensors)
        else:
            xs = self.body(tensor_list.tensors) #interm layer features
        if isinstance(xs, tuple):
            xs = torch.cat([xs[0], xs[1].permute(0,2,1)[:,:,:,None,None].expand(*xs[0].shape)], dim =1)
            xs = F.max_pool3d(xs, kernel_size=(4, 1, 1), stride=(4, 1, 1))
            xs = {'0' : self.mamba_proj(xs.permute(0,2,3,4,1).contiguous()).permute(0,4,1,2,3).contiguous()}

        if self.use_ViT:
            xs = self.space_forward(xs)

        out: Dict[str, NestedTensor] = {}

        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            mask = mask.unsqueeze(1).repeat(1,x.shape[2],1,1)
            out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_3d_backbone(cfg):
    position_embedding = build_position_encoding(cfg.CONFIG.MODEL.D_MODEL)
    backbone = Backbone(train_backbone=True, 
                     num_channels=cfg.CONFIG.MODEL.DIM_FEEDFORWARD, 
                     position_embedding=position_embedding, 
                     return_interm_layers=False,
                     cfg=cfg)
    model = Joiner(backbone, position_embedding)
    return model