import random

import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import sys
import scipy.io as scio
from torch.utils.data import Dataset
import torch


OS_NAME = sys.platform
SEP = '\\' if OS_NAME == "win32" else '/'

rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width, c):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    if c == 1:
        image_decoded = np.repeat(cv2.imread(filename, 0)[:, :, None], 3, axis=-1)
    else:
        image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class Video_DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1, c=1):
        self.c = c
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for idx, video in enumerate(sorted(videos)):
            video_name = video.split(SEP)[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['idx'] = idx
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split(SEP)[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):
                frames.append(self.videos[video_name]['frame'][i])

        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split(SEP)[-2]
        frame_name = int(self.samples[index].split(SEP)[-1].split('.')[-2])

        batch = []
        for i in range(self._time_step + self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                  self._resize_width, self.c)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0), self.transform(np.array([[self.videos[video_name]['idx']]]))

    def __len__(self):
        return len(self.samples)


class train_dataset(Dataset):
    """
    No data augmentation.
    Normalized from [0, 255] to [-1, 1], the channels are BGR due to cv2 and liteFlownet.
    """

    def __init__(self, cfg):
        self.img_h = cfg.img_size[0] # 256
        self.img_w = cfg.img_size[1] # 256
        self.clip_length = 5

        self.videos = []
        self.all_seqs = []
        for folder in sorted(glob.glob(f'{cfg.train_data}/*')):
            all_imgs = glob.glob(f'{folder}/*.jpg')
            all_imgs.sort()
            self.videos.append(all_imgs)

            random_seq = list(range(len(all_imgs) - 4))
            random.shuffle(random_seq)
            self.all_seqs.append(random_seq)

    def __len__(self):  # This decide the indice range of the PyTorch Dataloader.
        return len(self.videos)

    def __getitem__(self, indice):  # Indice decide which video folder to be loaded.

        one_folder = self.videos[indice]

        video_clip = []
        start = self.all_seqs[indice][-1]  # Always use the last index in self.all_seqs.

        for i in range(start, start + self.clip_length):
            video_clip.append(np_load_frame(one_folder[i], self.img_h, self.img_w))

        video_clip = np.array(video_clip).reshape((-1, self.img_h, self.img_w))
        video_clip = torch.from_numpy(video_clip)

        return indice, video_clip


class test_dataset:
    def __init__(self, cfg, video_folder):
        self.img_h = cfg.img_size[0]
        self.img_w = cfg.img_size[1]
        self.clip_length = 5
        self.imgs = glob.glob(video_folder + '/*.jpg')
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs) - (self.clip_length - 1)  # The first [input_num] frames are unpredictable.

    def __getitem__(self, indice):
        video_clips = []
        for frame_id in range(indice, indice + self.clip_length):
            video_clips.append(np_load_frame(self.imgs[frame_id], self.img_h, self.img_w))

        video_clips = np.array(video_clips).reshape((-1, self.img_h, self.img_w))
        video_clips = torch.from_numpy(video_clips) # new code
        return video_clips


class Label_loader:
    def __init__(self, cfg, video_folders):
        assert cfg.dataset_name in ('ped1', 'ped2', 'avenue', 'shanghai'), f'Did not find the related gt for \'{cfg.dataset}\'.'
        self.cfg = cfg
        self.name = cfg.dataset_name
        # self.frame_path = cfg.test_data
        self.mat_path = 'video/{}/{}.mat'.format(self.name, self.name)
        self.video_folders = video_folders

    def __call__(self):
        if self.name == 'shanghai':
            gt = self.load_shanghaitech()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.video_folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]

                sub_video_gt[start: end] = 1

            all_gt.append(sub_video_gt)

        return all_gt

    def load_shanghaitech(self):
        np_list = glob.glob(f'{self.cfg.data_root}/shanghai/testframemask/*_*.npy')
        np_list.sort()

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt


if __name__ == '__main__':
    import numpy

    print(numpy.__version__)

    import scipy

    print(scipy.__version__)
