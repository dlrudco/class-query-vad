"""
Class Query
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import pandas as pd
import cv2
import torch.utils.data as data
from glob import glob
import numpy as np
from utils.misc import collate_fn
import torch
import random
from PIL import Image
import torch.nn.functional as F
import datasets.video_transforms as T
import json
from utils.utils import print_log
import os
import csv

class VideoDataset(data.Dataset):

    def __init__(self, root_path, clip_len, frame_sample_rate,
                 transforms, crop_size=224, resize_size=256, mode="train", class_num=80, gpu_world_rank=0, log_path=None):
        self.frame_path = os.path.join(root_path, 'frames')

        self.annot_path = os.path.join(root_path, 'annotations', f'ava_{mode}_v2.2.csv')
        self.crop_size = crop_size
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.class_num = class_num
        self.resize_size = resize_size

        self.index_cnt = 0
        self._transforms = transforms
        self.mode = mode
        if gpu_world_rank == 0:
            print_log(log_path, "rescale size: {}, crop size: {}".format(resize_size, crop_size))
        self.read_ann_csv()
    #    breakpoint()
    
    def read_ann_csv(self):
        my_dict = dict()

        with open(self.annot_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                
                # Combine the first two columns to form the key
                key = '/'.join([row[0], row[1]])

                # Combine the next four columns to form the subkey
                subkey = '/'.join([row[2], row[3], row[4], row[5]])

                # Get the dictionary associated with the key, or create a new one if it doesn't exist
                sub_dict = my_dict.setdefault(key, dict())

                # Get the list associated with the subkey, or create a new one if it doesn't exist
                sub_list = sub_dict.setdefault(subkey, [])

                # Append the value to the sub-list
                sub_list.append(int(row[6]))
        # breakpoint()
        self.data_dict = my_dict
        self.data_list = list(my_dict.keys())
        self.data_len  = len(self.data_list)

    def __getitem__(self, index):
        vid, frame_second = self.data_list[index].split('/')
        timef = int(frame_second) - 900

        start_img = np.max((timef * 30 - self.clip_len // 2 * self.frame_sample_rate, 0))

        imgs, target = self.loadvideo(start_img, vid, self.data_list[index])

        if len(target) == 0 or target['boxes'].shape[0] == 0:
            pass
        else:
            if self._transforms is not None:
                imgs, target = self._transforms(imgs, target)

        while len(target) == 0 or target['boxes'].shape[0] == 0:
            print('resample.')
            self.index_cnt -= 1
            index = np.random.randint(len(self.data_list))
            vid, frame_second = self.data_list[index].split('/')
            timef = int(frame_second) - 900

            start_img = np.max((timef * 30 - self.clip_len // 2 * self.frame_sample_rate, 0))

            imgs, target = self.loadvideo(start_img, vid, self.data_list[index]) # ex) frame_key: '_eBah6c5kyA,1024' 

            if len(target)==0 or target['boxes'].shape[0] == 0:
                pass
            else:
                if self._transforms is not None:
                    imgs, target = self._transforms(imgs, target)

        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.permute(1, 0, 2, 3)

        return imgs, target

    def load_annotation(self, sample_id, video_frame_list): # (val 혹은 train의 key frame을 표시해놓은 list)

        num_classes = self.class_num
        boxes, classes = [], []
        target = {}

        first_img = cv2.imread(video_frame_list[0])

        oh = first_img.shape[0]
        ow = first_img.shape[1]
        if oh <= ow:
            nh = self.resize_size
            nw = self.resize_size * (ow / oh)
        else:
            nw = self.resize_size
            nh = self.resize_size * (oh / ow)

        p_t = int(self.clip_len // 2)
        key_pos = p_t
        # anno_entity = = self.data_dict[self.data_list[sample_id]]

        # for i, bbox in enumerate(anno_entity["bboxes"]):
        #     label_tmp = np.zeros((num_classes, ))
        #     acts_p = anno_entity["acts"][i]
        #     for l in acts_p:
        #         label_tmp[l] = 1

        #     if np.sum(label_tmp) == 0: continue
        #     p_x = np.int_(bbox[0] * nw)
        #     p_y = np.int_(bbox[1] * nh)
        #     p_w = np.int_(bbox[2] * nw)
        #     p_h = np.int_(bbox[3] * nh)

        #     boxes.append([p_t, p_x, p_y, p_w, p_h])
        #     classes.append(label_tmp)

        cur_frame_dict = self.data_dict[sample_id]
        for raw_bboxes in cur_frame_dict.keys():
            box = list(raw_bboxes.split('/'))
            box = [float(x) for x in box]
            box[0] *= nw
            box[1] *= nh
            box[2] *= nw
            box[3] *= nh

            label_tmp = np.zeros((num_classes, ))
            for x in cur_frame_dict[raw_bboxes]:
                label_tmp[x - 1] = 1
            
            boxes.append([p_t] + box)
            classes.append(label_tmp)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)
        boxes[:, 1::3].clamp_(min=0, max=int(nw))
        boxes[:, 2::3].clamp_(min=0, max=nh)

        if boxes.shape[0]:
            raw_boxes = F.pad(boxes, (1, 0, 0, 0), value=self.index_cnt)
        else:
            raw_boxes = boxes
        classes = np.array(classes)
        classes = torch.as_tensor(classes, dtype=torch.float32).reshape(-1, num_classes)

        target["image_id"] = [str(sample_id).replace(",", "_"), key_pos]
        target['boxes'] = boxes
        target['raw_boxes'] = raw_boxes
        target["labels"] = classes
        target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
        target["size"] = torch.as_tensor([int(nh), int(nw)])
        self.index_cnt = self.index_cnt + 1

        return target

    def loadvideo(self, start_img, vid, frame_key):
        # video_frame_path = self.frame_path.format(vid)
        video_frame_path = os.path.join(self.frame_path, vid)
        video_frame_list = sorted(glob(video_frame_path + '/*.jpg'))

        if len(video_frame_list) == 0:
            print("path doesnt exist", video_frame_path)
            return [], []

        target = self.load_annotation(frame_key, video_frame_list)

        start_img = np.max(start_img, 0)
        end_img = start_img + self.clip_len * self.frame_sample_rate
        indx_img = list(np.clip(range(start_img, end_img, self.frame_sample_rate), 0, len(video_frame_list) - 1))
        buffer = []
        for frame_idx in indx_img:
            tmp = Image.open(video_frame_list[frame_idx])
            tmp = tmp.resize((target['orig_size'][1], target['orig_size'][0]))
            buffer.append(tmp)

        return buffer, target

    def __len__(self):
        return self.data_len


def make_transforms(image_set, cfg):
    IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])}
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        print_log(log_path, "transform image crop: {}".format(cfg.CONFIG.DATA.IMG_SIZE))
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSizeCrop_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            T.ColorJitter(sat_shift=cfg.CONFIG.AUG.COLOR_JITTER,val_shift=cfg.CONFIG.AUG.COLOR_JITTER,),
            T.PCAJitter(alphastd=0.1,
                        eigval=IMAGENET_PCA['eigval'].numpy(),
                        eigvec=IMAGENET_PCA['eigvec'].numpy(),),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.Resize_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            normalize,
        ])

    if image_set == 'visual':
        return T.Compose([
            T.Resize_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')

def obtain_generated_bboxes_training(input_csv="../assets/ava_{}_v2.2.csv",
                                     eval_only=False,
                                     frame_root="/mnt/video-nfs5/datasets/ava/frames",
                                     mode="train"):
    import os
    from glob import glob
    used=[]
    input_csv = input_csv.format(mode)
    # frame_root = frame_root.format(mode)

    video_frame_bbox = {}
    gt_sheet = pd.read_csv(input_csv, header=None)
    count = 0
    frame_keys_list = set()
    missed_videos = set()

    for index, row in gt_sheet.iterrows():
        vid = row[0]
        if not os.path.isdir(frame_root + "/" + vid + ""):
            missed_videos.add(vid)
            continue

        frame_second = row[1]

        bbox_conf = row[7]
        if bbox_conf < 0.8:
            continue
        frame_key = "{},{}".format(vid, str(frame_second).zfill(4))

        frame_keys_list.add(frame_key)

        count += 1
        bbox = [row[2], row[3], row[4], row[5]]
        gt = int(row[6])

        if frame_key not in video_frame_bbox.keys():
            video_frame_bbox[frame_key] = {}
            video_frame_bbox[frame_key]["bboxes"] = [bbox]
            video_frame_bbox[frame_key]["acts"] = [[gt - 1]]
        else:
            if bbox not in video_frame_bbox[frame_key]["bboxes"]:
                video_frame_bbox[frame_key]["bboxes"].append(bbox)
                video_frame_bbox[frame_key]["acts"].append([gt - 1])
            else:
                idx = video_frame_bbox[frame_key]["bboxes"].index(bbox)
                video_frame_bbox[frame_key]["acts"][idx].append(gt - 1)

    print("missed vids:")
    print(missed_videos)
    return video_frame_bbox, list(frame_keys_list)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def build_dataloader(cfg, mode='val'):
    log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    if mode == 'train':
        train_dataset = VideoDataset(cfg.CONFIG.DATA.DATA_PATH,
                                transforms=make_transforms("train", cfg),
                                frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                clip_len=cfg.CONFIG.DATA.TEMP_LEN,
                                resize_size=cfg.CONFIG.DATA.IMG_SIZE,
                                crop_size=cfg.CONFIG.DATA.IMG_SIZE,
                                mode="train",
                                gpu_world_rank=cfg.DDP_CONFIG.GPU_WORLD_RANK,
                                log_path=log_path,)
        if cfg.DDP_CONFIG.DISTRIBUTED:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.CONFIG.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None),
        num_workers=9, sampler=train_sampler, pin_memory=True, collate_fn=collate_fn)
        if cfg.DDP_CONFIG.GPU_WORLD_RANK==0:
            print_log(log_path, "train anno is from:", train_loader.dataset.annot_path)
        return train_loader, train_sampler
    elif mode == 'val':
        val_dataset = VideoDataset(cfg.CONFIG.DATA.DATA_PATH,
                                transforms=make_transforms("val", cfg),
                                frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                clip_len=cfg.CONFIG.DATA.TEMP_LEN,
                                resize_size=cfg.CONFIG.DATA.IMG_SIZE,
                                crop_size=cfg.CONFIG.DATA.IMG_SIZE,
                                mode="val",
                                gpu_world_rank=cfg.DDP_CONFIG.GPU_WORLD_RANK,
                                log_path=log_path,)
        if cfg.DDP_CONFIG.DISTRIBUTED:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            val_sampler = None

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=False,
            num_workers=9, sampler=val_sampler, pin_memory=True, collate_fn=collate_fn)
        if cfg.DDP_CONFIG.GPU_WORLD_RANK==0:
            print_log(log_path, "val anno is from:", val_loader.dataset.annot_path)

        return val_loader, val_sampler
    else:
        raise ValueError("mode should be either train or val")

def reverse_norm(imgs):
    img = imgs
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = (img * std + mean) * 255.0
    img = img.transpose((1, 2, 0))[..., ::-1].astype(np.uint8)
    return img