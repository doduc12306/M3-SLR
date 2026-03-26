import os
import numpy as np
import sys
import torch
from torch.utils.data import  Dataset
import pandas as pd
from dataset.videoLoader import get_selected_indexs,pad_index
from decord import VideoReader
from utils.video_augmentation import *

class UFOneView_Dataset(Dataset):
    def __init__(self, base_url,split,dataset_cfg,train_labels = None,**kwargs):
        if train_labels is None:
            label_path = os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv")
            print("Label: ", label_path)
            self.train_labels = pd.read_csv(label_path, sep=',')
        else:
            print("Use labels from K-Fold")
            self.train_labels = train_labels

        # Normalize common CSV schemas into [video_name, label_id] columns.
        if 'file_name' in self.train_labels.columns and 'label_id' in self.train_labels.columns:
            self.train_labels = self.train_labels.rename(columns={'file_name': 'name', 'label_id': 'label'})
        elif 'video path' in self.train_labels.columns and 'gloss label ID' in self.train_labels.columns:
            self.train_labels = self.train_labels.rename(columns={'video path': 'name', 'gloss label ID': 'label'})

        if 'name' not in self.train_labels.columns or 'label' not in self.train_labels.columns:
            raise ValueError(
                "Label CSV must contain either [name,label], [file_name,label_id], or [video path,gloss label ID]."
            )

        print(split,len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.video_root = self._resolve_video_root(base_url, dataset_cfg)
        self.transform = self.build_transform(split)

    def _resolve_video_root(self, base_url, dataset_cfg):
        configured_video_folder = dataset_cfg.get('video_folder')
        if configured_video_folder:
            if os.path.isabs(configured_video_folder):
                return configured_video_folder
            return os.path.join(base_url, configured_video_folder)

        dataset_name = dataset_cfg.get('dataset_name', '')
        parent = os.path.dirname(base_url.rstrip('/\\'))
        candidates = [
            os.path.join(base_url, 'videos'),
            os.path.join(base_url, f'{dataset_name}_videos'),
            os.path.join(parent, f'{dataset_name}_videos'),
            os.path.join(base_url, 'MultiVSL200_videos'),
            os.path.join(parent, 'MultiVSL200_videos'),
            base_url,
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return os.path.join(base_url, 'videos')
        
    def build_transform(self, split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                        self.data_cfg['vid_transform']['NORM_STD_IMGNET'])
            )
        else:
            print("Build test/val transform")
            transform = Compose(
                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                        self.data_cfg['vid_transform']['NORM_STD_IMGNET'])
            )
        return transform
    def read_videos(self,name):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        clip = []
        path = name
        if not os.path.isabs(path):
            path = os.path.join(self.video_root, name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")

        vr = VideoReader(path,width=320, height=256)
        # print(path)
        sys.stdout.flush()
        vlen = len(vr)
        selected_index, pad = get_selected_indexs(vlen,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
        if pad is not None:
            selected_index  = pad_index(selected_index,pad).tolist()
        frames = vr.get_batch(selected_index).asnumpy()
        clip = []
        for frame in frames:
            clip.append(self.transform(frame))
            
        clip = torch.stack(clip,dim = 0)
        return clip

    def __getitem__(self, idx):
        self.transform.randomize_parameters()
        data = self.train_labels.iloc[idx].values

        name,label = data[0],data[1]

        clip = self.read_videos(name)

        # Keep output schema compatible with collate functions.
        return clip, str(name), torch.tensor(label)

    
    def __len__(self):
        return len(self.train_labels)
    
class UFThreeView_Dataset(Dataset):
    def __init__(self, base_url, split, dataset_cfg, **kwargs):
        label_path = os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv")
        print("Label: ", label_path)
        self.train_labels = pd.read_csv(label_path, sep=',')

        rename_map = {}
        if 'center_path' in self.train_labels.columns:
            rename_map['center_path'] = 'center'
        if 'left_path' in self.train_labels.columns:
            rename_map['left_path'] = 'left'
        if 'right_path' in self.train_labels.columns:
            rename_map['right_path'] = 'right'
        if 'label_id' in self.train_labels.columns:
            rename_map['label_id'] = 'label'
        if rename_map:
            self.train_labels = self.train_labels.rename(columns=rename_map)

        required_cols = {'center', 'left', 'right', 'label'}
        if not required_cols.issubset(set(self.train_labels.columns)):
            raise ValueError("Label CSV for three-view must contain [center,left,right,label].")

        print(split, len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.video_root = self._resolve_video_root(base_url, dataset_cfg)
        self.transform = self.build_transform(split)

    def _resolve_video_root(self, base_url, dataset_cfg):
        configured_video_folder = dataset_cfg.get('video_folder')
        if configured_video_folder:
            if os.path.isabs(configured_video_folder):
                return configured_video_folder
            return os.path.join(base_url, configured_video_folder)

        dataset_name = dataset_cfg.get('dataset_name', '')
        parent = os.path.dirname(base_url.rstrip('/\\'))
        candidates = [
            os.path.join(base_url, 'videos'),
            os.path.join(base_url, f'{dataset_name}_videos'),
            os.path.join(parent, f'{dataset_name}_videos'),
            os.path.join(base_url, 'MultiVSL200_videos'),
            os.path.join(parent, 'MultiVSL200_videos'),
            base_url,
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return os.path.join(base_url, 'videos')

    def build_transform(self, split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                        self.data_cfg['vid_transform']['NORM_STD_IMGNET'])
            )
        else:
            print("Build test/val transform")
            transform = Compose(
                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                        self.data_cfg['vid_transform']['NORM_STD_IMGNET'])
            )
        return transform

    def count_frames(self, video_path):
        vr = VideoReader(video_path, width=320, height=256)
        return len(vr)
    
    def read_one_view(self,name, selected_index):
        clip = []
        path = name
        if not os.path.isabs(path):
            path = os.path.join(self.video_root, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")

        vr = VideoReader(path,width=320, height=256)
        # print(path)
        # sys.stdout.flush()
        frames = vr.get_batch(selected_index).asnumpy()
        clip = []
        for frame in frames:
            clip.append(self.transform(frame))
            
        clip = torch.stack(clip,dim = 0)
        return clip

    def read_videos(self, center, left, right):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive', 'pad', 'central', 'pad'])

        center_path = center if os.path.isabs(center) else os.path.join(self.video_root, center)
        left_path = left if os.path.isabs(left) else os.path.join(self.video_root, left)
        right_path = right if os.path.isabs(right) else os.path.join(self.video_root, right)

        vlen1 = self.count_frames(center_path)
        vlen2 = self.count_frames(left_path)
        vlen3 = self.count_frames(right_path)

        min_vlen = min(vlen1, min(vlen2, vlen3))
        max_vlen = max(vlen1, max(vlen2, vlen3))
        if max_vlen - min_vlen < 10:
            selected_index, pad = get_selected_indexs(min_vlen - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_center = self.read_one_view(center, selected_index)

            rgb_left = self.read_one_view(left, selected_index)

            rgb_right = self.read_one_view(right, selected_index)
        else:
            selected_index, pad = get_selected_indexs(vlen1 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_center = self.read_one_view(center, selected_index)

            selected_index, pad = get_selected_indexs(vlen2 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_left = self.read_one_view(left, selected_index)

            selected_index, pad = get_selected_indexs(vlen3 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_right = self.read_one_view(right, selected_index)

        return rgb_left, rgb_center, rgb_right

    def __getitem__(self, idx):
        self.transform.randomize_parameters()
        row = self.train_labels.iloc[idx]
        center, left, right, label = row['center'], row['left'], row['right'], row['label']
        rgb_left, rgb_center, rgb_right= self.read_videos(center, left, right)

        return rgb_left, rgb_center, rgb_right, torch.tensor(label)

    def __len__(self):
        return len(self.train_labels)