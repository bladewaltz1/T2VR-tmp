import os
import random
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset

from config.base_config import Config
from datasets.video_capture import VideoCapture
from modules.basic_utils import load_json
from modules.tokenizer import clip_tokenizer, frequent_ids


class MSRVTTDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', img_transforms=None, max_len=77):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        db_file = 'data/MSRVTT/MSRVTT_data.json'
        test_csv = 'data/MSRVTT/MSRVTT_JSFUSION_test.csv'
        self.max_len = max_len

        if config.msrvtt_train_file == '7k':
            train_csv = 'data/MSRVTT/MSRVTT_train.7k.csv'
        else:
            train_csv = 'data/MSRVTT/MSRVTT_train.9k.csv'

        self.db = load_json(db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df['video_id'].unique()
            self._compute_vid2caption()
            self._construct_all_train_pairs()
            self.num_frames = self.config.num_frames
            self.num_test_frames = None
            self.video_sample_type = self.config.video_sample_type
        else:
            self.test_df = pd.read_csv(test_csv)
            self.num_frames = self.config.num_frames
            self.num_test_frames = self.config.num_test_frames
            self.video_sample_type = self.config.video_sample_type_test

    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.num_frames, 
                                                         self.num_test_frames,
                                                         self.video_sample_type)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        output = {'video_id': video_id, 'video': imgs, 'text': caption}

        if 'caption' in self.config.loss:
            masked_tokens = clip_tokenizer(caption)
            masked_tokens = masked_tokens['input_ids'][1:-1]
            selected_idx = []
            for i in range(len(masked_tokens)):
                if masked_tokens[i] not in frequent_ids:
                    if random.uniform(0, 1) < self.config.mask_ratio:
                        selected_idx.append(i)
                else:
                    if random.uniform(0, 1) < self.config.mask_ratio_freq:
                        selected_idx.append(i)

            mask = torch.zeros(self.max_len)
            mask[selected_idx] = 1
            output['mask'] = mask.bool()

        return output

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)

    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            caption = self.test_df.iloc[index].sentence

        return video_path, caption, vid

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption])

    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        for annotation in self.db['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)
