import torch
import numpy as np
import cv2
import time
from torch.utils.data import Dataset

# Using optical flow was my initial approach to the problem, but the feature correspondences from
# frame to frame were too inconsistent with my barebones Shi-Tomasi corner detection method to make
# the method successful. Therefore, none of the following code is used in the script, but is left
# here for reference.

class OpticalFlowDataset(Dataset):
    def __init__(self,*,indices):
        try:
            xTr = np.load('data/xTrOpt.npy')
        except IOError:
            xTr = self.dataset_from_video('data/train.mp4')
            np.save('data/xTrOpt.npy',xTr)
        yTr = np.loadtxt('data/train.txt')
        self.x = xTr[indices]
        self.y = yTr[indices]

    def __len__(self):
        return self.y.size

    def __getitem__(self, idx):
        # get features from previous frame, current frame, next frame
        # shape is (3, features_num, 2)
        x_padding = np.zeros_like(np.expand_dims(self.x[0], axis=0))
        padded_x = np.concatenate((x_padding.copy(), self.x, x_padding.copy()))
        feature_bundle = padded_x[idx:idx+3].flatten()    
        return (feature_bundle,self.y[idx])

    def dataset_from_video(self,filepath):
        start_time = time.time()
        # params for ShiTomasi corner detection
        feature_params = dict(  maxCorners = 20,
                                qualityLevel = 0.3,
                                minDistance = 7,
                                blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict(   winSize  = (15,15),
                            maxLevel = 2)
        train_cap = cv2.VideoCapture(filepath)
        frame_count = int(train_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        returned, first_frame = train_cap.read()
        first_gframe = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)

        features_list = []
        features_list.append(cv2.goodFeaturesToTrack(first_gframe, mask = None, **feature_params))

        frame_num=1
        returned = True
        prev_gframe = first_gframe
        while (frame_num < frame_count  and returned):
            if frame_num % 1000 == 0:
                print('Progress: {0:.1f}%'.format(frame_num/frame_count*100))
            returned, frame = train_cap.read()
            curr_gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            points, st, err = cv2.calcOpticalFlowPyrLK(prev_gframe, curr_gframe, features_list[frame_num-1], None, **lk_params)
            features_list.append(points)
            prev_gframe = curr_gframe
            print(st)
            frame_num += 1

        train_cap.release()
        features_tensor = np.array(features_list)
        features_tensor = np.squeeze(features_tensor)
        print("dataset_from_video completed in {} seconds.".format(time.time() - start_time))
        return features_tensor