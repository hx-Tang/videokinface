import random

import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
from PIL import Image


class kinfaceW(Dataset):
    def __init__(self, label):
        self.label = label

    def load_image(self, file):
        f = Image.open(file)
        f = f.resize((112,112))
        f = np.array(f)
        return f

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        p, c, l = self.label[index]
        p = self.load_image(p)
        c = self.load_image(c)
        p = torch.FloatTensor(p)
        c = torch.FloatTensor(c)
        return p, c, l


class UvA_NEMO(Dataset):
    def __init__(self, data_path, length=16):
        self.data_path = data_path
        self.length = length

    def load_video(self, file_path):
        file_path = self.data_path + '/' + file_path[:-4] + '_aligned.avi'
        videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
        videoreader.seek(0)
        vid_len = len(videoreader)
        indices = np.linspace(0, vid_len - 1, num=self.length)
        indices = np.clip(indices, 0, vid_len - 1).astype(np.int64)
        video = videoreader.get_batch(indices).asnumpy()
        return video

    def load_image(self, file_path):
        detail_path = self.data_path + '/' + file_path[:-4] + '.csv'
        detail = pd.read_csv(detail_path)
        idx = detail[' AU12_r'].idxmax()
        file_path = self.data_path + '/' + file_path[:-4] + '_aligned.avi'
        videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
        frame = videoreader[idx].asnumpy()
        return frame


class Contrastive_pretrain(UvA_NEMO):
    def __init__(self, data_path, label, m=2):
        self.data_path = data_path
        self.m = m
        self.label = label
        super().__init__(data_path)

    def __len__(self):
        return len(self.label)

    def load_video_clips(self, file_path, idx):
        file_path = self.data_path + '/' + file_path[:-4] + '_aligned.avi'
        videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
        indices = np.arange(idx-self.m,self.m+idx+1).astype(np.int64)
        clip = videoreader.get_batch(indices).asnumpy()
        return clip

    def __getitem__(self, index):
        q_name, idx, a_name, a_idx, intra_n_idx, n_name, n_idx = self.label[index]

        q = self.load_video_clips(q_name, int(idx))
        a = self.load_video_clips(a_name, int(a_idx))
        n = self.load_video_clips(n_name, int(n_idx))
        n_intra = self.load_video_clips(a_name, int(intra_n_idx))

        q = torch.FloatTensor(q)
        a = torch.FloatTensor(a)
        n = torch.FloatTensor(n)
        n_intra = torch.FloatTensor(n_intra)

        return q, a, n, n_intra


class Pretrained(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        a = self.label[index]
        anchor = self.load_video(a)
        anchor = self.transform(list(anchor), return_tensors="pt")
        anchor = anchor['pixel_values'].squeeze()
        return anchor


class Age(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform
        self.subject2details = np.load('protocols/subject2details.npy', allow_pickle=True).item()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        a = self.label[index]
        age = self.subject2details[a[:3]]['age']/100
        age = torch.FloatTensor([age])
        anchor = self.load_video(a)
        anchor = self.transform(list(anchor), return_tensors="pt")
        anchor = anchor['pixel_values'].squeeze()
        return anchor, age


class VideoClassification(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16, cnn=False):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform
        self.cnn = cnn

    def __len__(self):
        return len(self.label)

    def cat_videos(self, vid1, vid2):
        if self.transform:
            vid1 = self.transform(list(vid1), return_tensors="pt")
            vid1 = vid1['pixel_values'].squeeze()
            vid2 = self.transform(list(vid2), return_tensors="pt")
            vid2 = vid2['pixel_values'].squeeze()
        vid_cat = [vid1,vid2]
        random.shuffle(vid_cat)
        return vid_cat

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_video(a)
        pos = self.load_video(p)
        neg = self.load_video(n)

        if torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, pos)
            label = torch.LongTensor([1])
        elif torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, neg)
            label = torch.LongTensor([0])
        else:
            anchor = self.cat_videos(pos, neg)
            label = torch.LongTensor([0])
        return anchor, label


class VideoClassificationCat(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16, cnn=False):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform
        self.cnn = cnn

    def __len__(self):
        return len(self.label)

    def cat_videos(self, vid1, vid2):
        vid1 = self.transform(list(vid1), return_tensors="pt")
        vid1 = vid1['pixel_values'].squeeze()
        vid2 = self.transform(list(vid2), return_tensors="pt")
        vid2 = vid2['pixel_values'].squeeze()
        vid_cat = [vid1,vid2]
        random.shuffle(vid_cat)
        vid_cat = torch.cat(vid_cat, 1)
        return vid_cat

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_video(a)
        pos = self.load_video(p)
        neg = self.load_video(n)

        if torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, pos)
            label = torch.LongTensor([1])
        elif torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, neg)
            label = torch.LongTensor([0])
        else:
            anchor = self.cat_videos(pos, neg)
            label = torch.LongTensor([0])
        return anchor, label


class Triplet(UvA_NEMO):
    def __init__(self, data_path, label, transform=None, length=16, cnn=False):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform
        self.cnn = cnn

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_video(a)
        pos = self.load_video(p)
        neg = self.load_video(n)

        # anchor = self.transform(list(anchor), return_tensors="pt")
        # pos = self.transform(list(pos), return_tensors="pt")
        # neg = self.transform(list(neg), return_tensors="pt")

        anchor = torch.FloatTensor(anchor)
        pos = torch.FloatTensor(pos)
        neg = torch.FloatTensor(neg)

        # if self.cnn:
        #     anchor = anchor['pixel_values'].squeeze().permute(1, 0, 2, 3)
        #     pos = pos['pixel_values'].squeeze().permute(1, 0, 2, 3)
        #     neg = neg['pixel_values'].squeeze().permute(1, 0, 2, 3)
        # else:
        #     anchor = anchor['pixel_values'].squeeze()
        #     pos = pos['pixel_values'].squeeze()
        #     neg = neg['pixel_values'].squeeze()

        return anchor, pos, neg

class Image_pairs(UvA_NEMO):
    def __init__(self, data_path, label, transform=None, length=16):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_image(a)
        pos = self.load_image(p)
        neg = self.load_image(n)

        anchor = self.transform(anchor, return_tensors="pt")
        pos = self.transform(pos, return_tensors="pt")
        neg = self.transform(neg, return_tensors="pt")

        anchor = anchor['pixel_values'].squeeze()
        pos = pos['pixel_values'].squeeze()
        neg = neg['pixel_values'].squeeze()

        return anchor, pos, neg


class ImageClassification(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def cat_videos(self, vid1, vid2):
        vid1 = self.transform(vid1, return_tensors="pt")
        vid1 = vid1['pixel_values'].squeeze()
        vid2 = self.transform(vid2, return_tensors="pt")
        vid2 = vid2['pixel_values'].squeeze()
        vid_cat = [vid1,vid2]
        random.shuffle(vid_cat)
        return vid_cat

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_image(a)
        pos = self.load_image(p)
        neg = self.load_image(n)

        if torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, pos)
            label = torch.LongTensor([1])
        elif torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, neg)
            label = torch.LongTensor([0])
        else:
            anchor = self.cat_videos(pos, neg)
            label = torch.LongTensor([0])
        return anchor, label





if __name__ == '__main__':
    data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'
    fold = 1
    group = 'wholeset/spontaneous'
    train_label = np.load('./protocols/' + group + '/train/' + str(fold) + '.npy')
    dataset = Image_pairs(data_path, train_label)
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=1)
    for f in loader:
        print(f[0].shape)