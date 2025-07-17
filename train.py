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
from utils.model_utils import deploy_model, load_model_and_states, save_checkpoint
from utils.video_action_recognition import validate
from pipelines.video_action_recognition_config import get_cfg_defaults
from pipelines.launch import spawn_workers
from utils.utils import print_log
import random
import os
import wandb

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
    # cfg.CONFIG.MODEL.LOAD_DETR = cfg.CONFIG.MODEL.LOAD_DETR

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
    train_loader, _ = build_dataloader(cfg, mode='train')
    val_loader, _ = build_dataloader(cfg, mode='val')

    # create criterion
    criterion = criterion.cuda()
    
    if cfg.CONFIG.AMP:
        scaler = torch.cuda.amp.GradScaler(growth_interval=1000)
    else:
        scaler = None

    model = load_model_and_states(model, scaler, cfg)
    start_time = time.time()
    
    start_epoch = 0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.CONFIG.TRAIN.LR))
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        exp_name = cfg.CONFIG.LOG.EXP_NAME
        logger = wandb.init(project='class_query_vad', config=cfg, name=exp_name)
    else:
        logger = None
    
    best_Map = 0
    for cur_epoch in range(start_epoch, cfg.CONFIG.TRAIN.NUM_EPOCHS):
        avg_loss = train_epoch(cfg, model, criterion, optimizer, scaler, postprocessors, train_loader, f'cuda:{cfg.DDP_CONFIG.GPU_WORLD_RANK}', cur_epoch, logger=logger)
        if (cur_epoch + 1) % cfg.CONFIG.TRAIN.get('EVAL_FREQ', 1) == 0:
            model.eval()
            with torch.no_grad():
                _Map, metrics = validate(cfg, model, criterion, postprocessors, val_loader, cur_epoch)
            if logger is not None:
                logger.log(metrics, commit=False)
            if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
                if best_Map < _Map:
                    best_Map = _Map
                save_checkpoint(cfg, cur_epoch, model, _Map, best_Map, optimizer, scaler)

        if logger is not None:
            logger.log({'epoch' : cur_epoch, 'avg_loss' : avg_loss}, commit=True)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0: 
        print_log(save_path, 'Epoch time {}'.format(total_time_str))

class ExpAverageMeter():
    def __init__(self, beta=0.9):
        self.beta = beta
        self.val = None
        self.avg = None
        self.count = 0
    def update(self, val):
        self.val = val
        self.avg = self.beta * self.avg + (1 - self.beta) * val if self.avg is not None else val
        self.count += 1
    def reset(self):
        self.val = None
        self.avg = None
        self.count = 0

def train_epoch(cfg, model, criterion, optimizer, scaler, postprocessor, dataloader, device, epoch, logger=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    total_updates_per_epoch = (len(dataloader) + cfg.CONFIG.TRAIN.GRAD_ACCUM - 1) // cfg.CONFIG.TRAIN.GRAD_ACCUM
    total_update = 0
    batch_time = time.time()
    avg_meters = {}
    batch_loss = 0.
    for step, batch in enumerate(dataloader, 1):
        # Assume batch is a tuple of (inputs, targets)
        inputs, targets = batch
        inputs = inputs.to(device)
        # Move target tensors to the device if needed
        batch_id = [t["image_id"] for t in targets]
        for t in targets:
            del t["image_id"]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 
        with torch.cuda.amp.autocast(enabled=cfg.CONFIG.AMP):
            outputs = model(inputs)  # Forward pass
            loss_dict = criterion(outputs, targets) # Compute losses
            # Sum weighted losses (using loss weights from the criterion)
            losses = sum(loss_dict[k] * w for k, w in criterion.weight_dict.items())
            batch_loss += losses.item() / cfg.CONFIG.TRAIN.GRAD_ACCUM
        if scaler is None:
            losses.backward()
        else:
            scaler.scale(losses).backward()
        for key, weight in criterion.weight_dict.items():
            if key not in avg_meters:
                avg_meters[key] = ExpAverageMeter()
            avg_meters[key].update(loss_dict[key].item() * weight)
        if (step+1) % cfg.CONFIG.TRAIN.GRAD_ACCUM == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            total_update += 1 
            
            if total_update % cfg.CONFIG.TRAIN.PRINT_INTERVAL == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
                print(f"Epoch [{epoch}] Step [{total_update}/{total_updates_per_epoch}] Loss: {batch_loss:.4f} Time: {time.time() - batch_time:.2f}s")
                if logger is not None:
                    loss_dict_reduced_scaled = {k: avg_meters[k].avg
                                        for k, v in criterion.weight_dict.items()}
                    logger.log({'total_step' : epoch * total_updates_per_epoch + total_update, 'total_loss' : batch_loss, **loss_dict_reduced_scaled})
                batch_time = time.time()
            batch_loss = 0.
        total_loss += losses.item()
        
            
    
    return total_loss / len(dataloader)

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
    parser.add_argument('--root_data_path', type=str)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    
    now = datetime.datetime.now()
    study = now.strftime("%Y-%m-%d")
    run = now.strftime("%H-%M")

    cfg.CONFIG.DATA.DATA_PATH = args.root_data_path
    cfg.CONFIG.DATA.LABEL_PATH = os.path.join(args.root_data_path, cfg.CONFIG.DATA.LABEL_PATH)
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
