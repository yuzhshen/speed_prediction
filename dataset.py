import torch
import numpy as np
import cv2
import time
from torch.utils.data import Dataset

class ImageDatasetTrain(Dataset):
    def __init__(self,*,indices):
        try:
            xTr = np.load('data/xTr.npy')
        except IOError:
            xTr = self.features_from_video('data/train.mp4')
            np.save('data/xTr.npy',xTr)
        yTr = np.loadtxt('data/train.txt')
        xTr = xTr[indices]
        x_padding = np.zeros_like(np.expand_dims(xTr[0], axis=0))
        self.x = np.concatenate((x_padding, xTr, x_padding))
        self.y = yTr[indices]

    def features_from_video(self,filepath):
        """ Returns ndarray of shape (len, 3, 120, 160)
        """
        start_time = time.time()
        VID_WIDTH = 640
        VID_HEIGHT = 480
        train_cap = cv2.VideoCapture(filepath)
        frame_count = int(train_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        features_list = []
        frame_num=0
        returned = True
        while (frame_num < frame_count  and returned):
            returned, frame = train_cap.read()
            resized = cv2.resize(frame, (int(VID_WIDTH/4),int(VID_HEIGHT/4)))
            resized = np.swapaxes(resized,0,2)
            channel_first = np.swapaxes(resized,1,2)
            features_list.append(channel_first)
            frame_num += 1
            if frame_num % 1000 == 0:
                print('features_from_video progress: {0:.1f}%'.format(frame_num/frame_count*100))
        train_cap.release()
        features_tensor = np.array(features_list)
        print("features_from_video completed in {} seconds.".format(time.time() - start_time))
        return features_tensor

    def __len__(self):
        return self.y.size

    def __getitem__(self, idx):
        # get features from previous 2 frames, current frame, next 2 frames
        # shape is (5, height, width)
        features = torch.tensor(self.x[idx:idx+3], dtype=torch.float)
        label = torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(0)
        return (features,label)

class ImageDatasetTest(ImageDatasetTrain):
    def __init__(self):
        try:
            xTr = np.load('data/xTe.npy')
        except IOError:
            xTr = self.features_from_video('data/test.mp4')
            np.save('data/xTe.npy',xTr)
        x_padding = np.zeros_like(np.expand_dims(xTr[0], axis=0))
        self.x = np.concatenate((x_padding, xTr, x_padding))

    def __len__(self):
        return self.x.shape[0]-2

    def __getitem__(self, idx):
        # get features from previous 2 frames, current frame, next 2 frames
        # shape is (5, height, width)
        features = torch.tensor(self.x[idx:idx+3], dtype=torch.float)
        return features