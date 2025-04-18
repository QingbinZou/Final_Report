import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from skimage.util import random_noise

class RSDataset(Sequence):
    def __init__(self, rs_dir, gs_dir, size=(768, 1024), batch_size=1, shuffle=True, augment=True, mode='train'):
        self.rs_dir = rs_dir
        self.gs_dir = gs_dir
        self.size = size  # (height, width)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode.lower()
        self.files = sorted(os.listdir(rs_dir))
        self.on_epoch_end()

    def __len__(self):
        return len(self.files) // self.batch_size

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        x, y = [], []
        for f in batch_files:
            rs = self.load_image(os.path.join(self.rs_dir, f), is_rs=True)
            gs = self.load_image(os.path.join(self.gs_dir, f), is_rs=False)
            x.append(rs)
            y.append(gs)
        return np.array(x), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.files)

    def load_image(self, path, is_rs):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像：{path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width = self.size
        img = cv2.resize(img, (width, height)).astype(np.float32) / 255.0

        # 仅在训练模式且为 RS 输入时进行数据增强
        if self.mode == 'train' and is_rs and self.augment:
            for c in range(3):
                img[..., c] = cv2.GaussianBlur(img[..., c], (5, 3), 1.5)
                img[..., c] = random_noise(img[..., c], mode='gaussian', var=0.001)
            img = np.clip(img, 0, 1).astype(np.float32)

        return img
