import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import time
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self,*,indices):
        try:
            xTr = np.load('data/xTr.npy')
        except IOError:
            xTr = self.dataset_from_video('data/train.mp4')
            np.save('data/xTr.npy',xTr)
        yTr = np.loadtxt('data/train.txt')
        xTr = xTr[indices]
        x_padding = np.zeros_like(np.expand_dims(xTr[0], axis=0))
        self.x = np.concatenate((x_padding, x_padding, xTr, x_padding, x_padding))
        self.y = yTr[indices]

    def __len__(self):
        return self.y.size

    def __getitem__(self, idx):
        # get features from previous 2 frames, current frame, next 2 frames
        # shape is (5, height, width)
        features = torch.tensor(self.x[idx:idx+5], dtype=torch.float)
        label = torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(0)
        return (features,label)
    
    def dataset_from_video(self, filepath):
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
            gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gframe, (int(VID_WIDTH/4),int(VID_HEIGHT/4)))
            resized = np.squeeze(resized)
            features_list.append(resized)
            frame_num += 1
            if frame_num % 1000 == 0:
                print('Progress: {0:.1f}%'.format(frame_num/frame_count*100))
        train_cap.release()
        features_tensor = np.array(features_list)
        print("dataset_from_video completed in {} seconds.".format(time.time() - start_time))
        return features_tensor

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(5,10,5)
        self.conv2 = nn.Conv2d(10,20,5)
        self.conv3 = nn.Conv2d(20,5,3)
        self.fc1 = nn.Linear(25*35*5,50)
        self.fc2 = nn.Linear(50,10)
        self.fc3 = nn.Linear(10,1)
    
    def forward(self, x):
        # x is (batch,120)
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, np.prod(x.shape[1:]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    MAX_EPOCHS = 15
    BATCH_SIZE = 32
    BATCHES_PER_REPORT = 20
    MODEL_PATH = 'models/conv_model_decay_1_shuffled'

    cap = cv2.VideoCapture('data/train.mp4')
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # seed set for reproducibility
    torch.manual_seed(7)
    
    val_idx = int(0.9*FRAME_COUNT)
    tr_data = ImageDataset(indices=list(range(0,val_idx)))
    val_data = ImageDataset(indices=list(range(val_idx,FRAME_COUNT)))
    tr_generator = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True)
    val_generator = DataLoader(val_data, batch_size=FRAME_COUNT-val_idx)
    net = Model()
    try:
        net.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        optimizer = optim.Adam(net.parameters(),weight_decay=0.1)
        criterion = nn.MSELoss()
        for epochs in range(MAX_EPOCHS):
            running_loss = 0
            for enum, batch in enumerate(tr_generator,1):
                optimizer.zero_grad()
                inpt, labels = batch[0], batch[1]
                output = net(inpt)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                if enum%BATCHES_PER_REPORT == 0:
                    print('Epoch: {} - Progress: {:.2f}% - MSE loss: {}'
                        .format(epochs+1, enum*BATCH_SIZE/val_idx*100,running_loss/BATCHES_PER_REPORT))
                    running_loss = 0
            for batch in val_generator:
                inpt, labels = batch[0], batch[1]
                output = net(inpt)
                val_loss = criterion(output, labels)
                print('Validation loss: {}'.format(val_loss))
        torch.save(net.state_dict(), MODEL_PATH)

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