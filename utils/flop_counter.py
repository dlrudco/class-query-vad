"""
Class Query
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
from pipelines.video_action_recognition_config import get_cfg_defaults
from models.model import build_model
import fvcore.nn as fv


cfg = get_cfg_defaults()
cfg.merge_from_file("./configuration/config-file.yaml")
model, _, _ = build_model(cfg)

device = "cuda:0"
model = model.to(device)

inp = torch.randn((1, 3, 16, 256, 320)).to(device)


def get_FLOPs_params(model, x):
    def _human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return '%.3f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
            
    out = fv.FlopCountAnalysis(model, x)
    params = fv.parameter_count(model)
    params_table = fv.parameter_count_table(model)
    flops_human = _human_format(out.total())
    params_human =  _human_format(int(params['']))
    
    return flops_human, params_human

flops, params = get_FLOPs_params(model, inp)
print("flops:", flops)
print("params:", params)