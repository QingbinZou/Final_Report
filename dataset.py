import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from skimage.util import random_noise

class RSDataset(Sequence):
    def __init__(self, rs_dir, gs_dir, size=(384, 384), batch_size=4, shuffle=True, augment=True):
        self.rs_dir = rs_dir
        self.gs_dir = gs_dir
        self.size = size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
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
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.size).astype(np.float32) / 255.0

        # 对 RS 图像进行模糊增强（仅训练时启用）
        if is_rs and self.augment:
            img = cv2.GaussianBlur(img, (5, 3), 1.5)
            img = random_noise(img, mode='gaussian', var=0.001)
            img = np.clip(img, 0, 1).astype(np.float32)

        return np.expand_dims(img, axis=-1)
