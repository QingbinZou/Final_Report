import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from config import IMAGE_SIZE, NUM_CHANNELS, STACK_SIZE

class AlignedRSDataset(Sequence):
    def __init__(self, rs_root, gs_root, size=IMAGE_SIZE, batch_size=4, shuffle=True, augment=False):
        self.rs_root = rs_root
        self.gs_root = gs_root
        self.size = size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.index_tuples = self._make_index()
        self.on_epoch_end()

    def _make_index(self):
        index = []
        folders = sorted(os.listdir(self.rs_root))
        for folder in folders:
            for i in range(1, 49):  # 中间帧编号 1~48
                index.append((folder, i))
        return index

    def __len__(self):
        return len(self.index_tuples) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.index_tuples[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch, y_batch = [], []
        for folder, i in batch_idx:
            prev = self._load_rgb(self.rs_root, folder, i - 1)
            curr = self._load_rgb(self.rs_root, folder, i)
            next = self._load_rgb(self.rs_root, folder, i + 1)
            gt = self._load_rgb(self.gs_root, folder, i)

            if self.augment:
                prev, curr, next, gt = self._random_flip(prev, curr, next, gt)

            stacked = np.concatenate([prev, curr, next], axis=-1)
            x_batch.append(stacked)
            y_batch.append(gt)

        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_tuples)

    def _load_rgb(self, root, folder, idx):
        path = os.path.join(root, folder, f"{idx:08d}.png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.size)
        img = img.astype(np.float32) / 255.0
        return img

    def _random_flip(self, prev, curr, next, gt):
        if np.random.rand() > 0.5:
            prev = np.fliplr(prev)
            curr = np.fliplr(curr)
            next = np.fliplr(next)
            gt = np.fliplr(gt)
        return prev, curr, next, gt
