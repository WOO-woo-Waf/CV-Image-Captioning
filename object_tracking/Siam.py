import cv2
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import torch

class SiameNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96, 256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256, 384, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3)
        )
    
    def forward(self, x):
        return self.features(x)
    
def cross_relation(z, x):
    # 可以理解是较小的模板在较大的图像上滑动，下面操作是每个样本的每个通道分别做互相关
    B, C, HZ, WZ = z.size()
    _, _, HX, WX = x.size()
    z = z.view(B*C, 1, HZ, WZ)
    x = x.view(1, B*C, HX, WX)
    out = nn.functional.conv2d(x, z, groups=B*C)
    out = out.view(B, C, out.size(-2, out.size(-1)))
    return out.sum(dim=1, keepdim=True)

class TrackingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = [] # 这里面存的是video对应的每一帧
        self.videos = []

        for vid in os.listdir(root_dir):
            vid_dir = os.path.join(root_dir, vid)
            gt_file = os.path.join(vid_dir, 'groundtruth.txt')
            img_files = [f for f in os.listdir(vid_dir) if f.endswith('.jpg')]
            if img_files and os.path.exists(gt_file):
                self.videos.append(vid_dir)
                num_frames = len([f for f in img_files if f.endswith('.jpg')])
                for i in range(1, num_frames):
                    self.pairs.append((vid_dir,0, i))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vid, frm0, frm = self.pairs[idx]
        img_files = sorted([f for f in os.listdir(vid) if f.endswith('.jpg')])
        gt_file = os.path.join(vid, 'groundtruth.txt')

        with open(gt_file, 'r') as f:
            gt_lines = f.readlines()

        bbox0 = np.array(gt_lines[frm0].strip().split(','), dtype=np.float32)
        x0, y0, w0, h0 = bbox0
        bbox1 = np.array(gt_lines[frm].strip().split(','), dtype=np.float32)
        x, y, w, h = bbox1
        frame0 = cv2.imread(os.path.join(vid, img_files[frm0]))
        frame = cv2.imread(os.path.join(vid, img_files[frm]))
        z = frame0[int(y0):int(y0+h0), int(x0):int(x0+w0)]
        cx, cy = x + w/2, y + h/2
        size = int(max(w, h) * 2)
        x1 = int(cx - size/2); y1 = int(cy - size/2)
        x2 = x1 + size; y2 = y1 + size
        crop = lambda img, x1, y1, x2, y2: img[max(0,y1):y2, max(0,x1):x2]
        x = crop(frame, x1, y1, x2, y2)
        # 因为siamfc不支持多尺寸，所以得手动resize成设定好的
        z = cv2.resize(z, (127, 127))
        x = cv2.resize(x, (255, 255))
        if self.transform:
            z = self.transform(z)
            x = self.transform(x)
        H, W = 17, 17 
        sigma = 1.0
        y_coords = torch.arange(H, dtype=torch.float32)
        x_coords = torch.arange(W, dtype=torch.float32)
        y_coords, x_coords = torch.meshgrid(y_coords, x_coords, indexing='ij')
        center_y, center_x = H/2, W/2
        gaussian = torch.exp(-((y_coords - center_y)**2 + (x_coords - center_x)**2) / (2*sigma**2))
        label_map = gaussian / gaussian.sum()
        return z, x, label_map
