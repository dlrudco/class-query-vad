"""
Class Query
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import datetime
import time

import torch
import torch.optim
from models.model import build_model
from utils.model_utils import deploy_model, load_model_and_states
from utils.video_action_recognition import validate
from pipelines.video_action_recognition_config import get_cfg_defaults
from pipelines.launch import spawn_workers
from utils.utils import print_log
import random
import os

# Function to write or append the this_ip to the file
def write_this_ip_to_file(file_path, this_ip):
    with open(file_path, 'a') as file:
        file.write(this_ip + '\n')

# Function to read the file and return a list of IPs
def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        ip_list = file.read().splitlines()
    return ip_list

def main_worker(cfg):
    
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        save_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)

    cfg.CONFIG.MODEL.LOAD = True
    cfg.CONFIG.MODEL.LOAD_FC = True
    cfg.CONFIG.MODEL.LOAD_DETR = False

    # create model
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:    
        print_log(save_path, datetime.datetime.today())
        print_log(save_path, 'Creating the model: %s' % cfg.CONFIG.MODEL.NAME)
        print_log(save_path, "use single frame:", cfg.CONFIG.MODEL.SINGLE_FRAME)
    model, criterion, postprocessors = build_model(cfg)
    model = deploy_model(model, cfg)
    num_parameters = sum(p.numel() for p in model.parameters())
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:    
        print_log(save_path, 'Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))

    if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
        from datasets.ava_frame import build_dataloader
    elif cfg.CONFIG.DATA.DATASET_NAME == 'jhmdb':
        from datasets.jhmdb_frame import build_dataloader
    elif cfg.CONFIG.DATA.DATASET_NAME == 'ucf':
        from datasets.ucf_frame import build_dataloader        
    else:
        build_dataloader = None
        print("invalid dataset name; dataset name should be one of 'ava, 'jhmdb', or 'ucf'.")

    print(f"writing results in {cfg.CONFIG.LOG.BASE_PATH}/{cfg.CONFIG.LOG.EXP_NAME}/...")
    
    # create dataset and dataloader
    val_loader, _, = build_dataloader(cfg)

    # create criterion
    criterion = criterion.cuda()
    
    if cfg.CONFIG.AMP:
        scaler = torch.cuda.amp.GradScaler(growth_interval=1000)
    else:
        scaler = None

    model = load_model_and_states(model, scaler, cfg)

    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0: 
        print_log(save_path, 'Start evaluating...')
    start_time = time.time()
    
    epoch = 0
    validate(cfg, model, criterion, postprocessors, val_loader, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0: 
        print_log(save_path, 'Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='./configuration/AVA22_CSN152.yaml',
                        help='path to config file.')
    parser.add_argument('--debug', action='store_true', help="debug, and ddp is disabled")
    parser.add_argument('--amp', action='store_true', help="use average mixed precision")
    parser.add_argument('--split', default=0, type=int, help="dataset split (for jhmdb)")
    parser.add_argument('--pretrained_path',
                        default='',
                        help='path to pretrained .pth file')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    
    now = datetime.datetime.now()
    study = now.strftime("%Y-%m-%d")
    run = now.strftime("%H-%M")

    cfg.CONFIG.LOG.RES_DIR = cfg.CONFIG.LOG.RES_DIR.format(study, run)
    cfg.CONFIG.LOG.EXP_NAME = cfg.CONFIG.LOG.EXP_NAME.format(study, run)
    if args.debug:
        cfg.DDP_CONFIG.DISTRIBUTED = False
        cfg.CONFIG.LOG.RES_DIR = "debug_{}-{}/res/".format(study,run)
        cfg.CONFIG.LOG.EXP_NAME = "debug_{}-{}".format(study,run)
    if args.amp:
        cfg.CONFIG.AMP = True          
    if cfg.CONFIG.DATA.DATASET_NAME == 'jhmdb':
        cfg.CONFIG.DATA.SPLIT = args.split
        if args.split in [1,2]:
            cfg.CONFIG.LOG.EXP_NAME = cfg.CONFIG.LOG.EXP_NAME + f"_{args.split}"
            cfg.CONFIG.LOG.RES_DIR = cfg.CONFIG.LOG.EXP_NAME + "/res"
            cfg.DDP_CONFIG.DIST_URL = ":".join(cfg.DDP_CONFIG.DIST_URL.split(":")[:2]) + ":" + str(args.split+11111+random.randint(0,9))
    cfg.CONFIG.MODEL.PRETRAINED_PATH = args.pretrained_path
    
    import socket 
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    this_ip = s.getsockname()[0] # put this to world_url

    if cfg.DDP_CONFIG.WORLD_SIZE > 1:
        tmp_path = '{}/ip_lists/{}-{}.txt'
        file_path = tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, study, run)
        if not os.path.exists(file_path):    
            with open(file_path, 'w') as f:
                f.write(this_ip + '\n')
        else:
            write_this_ip_to_file(file_path, this_ip)
            
        while True:
            ip_lines = read_file_to_list(file_path)
            if len(ip_lines) == cfg.DDP_CONFIG.WORLD_SIZE:
                break
            time.sleep(0.5)
        
        ip_list = read_file_to_list(file_path)
        cfg.DDP_CONFIG.WORLD_URLS = ip_list
        cfg.DDP_CONFIG.DIST_URL = cfg.DDP_CONFIG.DIST_URL.format(ip_list[0])        
        
    else:    
        cfg.DDP_CONFIG.DIST_URL = cfg.DDP_CONFIG.DIST_URL.format(this_ip)
        cfg.DDP_CONFIG.WORLD_URLS[0] = cfg.DDP_CONFIG.WORLD_URLS[0].format(this_ip)

    s.close()
    spawn_workers(main_worker, cfg)
