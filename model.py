import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
from torch.utils.data import DataLoader
import dataset as d
import sys

class ModifiedVGG(nn.Module):
    def __init__(self):
        super(ModifiedVGG,self).__init__()
        vgg_model = torchvision.models.vgg11_bn(pretrained=True)
        for param in vgg_model.parameters():
            param.requires_grad = False	
        self.conv = nn.Sequential(*list(vgg_model.features.children())[0:29])
        self.gru = nn.GRU(input_size=7680, hidden_size=50, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(300, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(100,1))
 	    
    def forward(self,x):
        pre_fc = []
        for idx in range(3):
            localx = self.conv(x[:,idx,:,:,:])
            localx = localx.view(-1, 1, np.prod(localx.shape[1:]))
            pre_fc.append(localx)
        x = torch.cat(pre_fc,dim=1)
        x = self.gru(x)[0]
        x = x.contiguous().view(-1, np.prod(x.shape[1:]))
        x = self.fc(x)
        return x

def get_num_frames(filepath):
    cap = cv2.VideoCapture(filepath)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frames

if __name__ == '__main__':
    EPOCH_NUM = int(sys.argv[1])
    BATCH_SIZE = 32
    BATCHES_PER_REPORT = 20
    MODEL_FOLDER = sys.argv[2]
    MODEL_PATH = MODEL_FOLDER+'/vgg_model_e'+str(EPOCH_NUM)
    PREV_MODEL_PATH = MODEL_FOLDER+'/vgg_model_e'+str(EPOCH_NUM-1)
    FRAME_COUNT = get_num_frames('data/train.mp4')

    cuda = torch.device('cuda')

    print('Creating datasets...')
    val_idx = int(FRAME_COUNT)
    tr_data = d.ImageDatasetTrain(indices=list(range(0,val_idx)))
    te_data = d.ImageDatasetTest()
    val_data = d.ImageDatasetTrain(indices=list(range(val_idx,FRAME_COUNT)))
    tr_generator = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True)
    te_generator = DataLoader(te_data)
    val_generator = DataLoader(val_data)
    net = ModifiedVGG()
    net.to(cuda)
    optimizer = optim.Adam(net.parameters(),weight_decay=0.1)
    criterion = nn.MSELoss()
    try:
        net.load_state_dict(torch.load(MODEL_PATH))
        # validation accuracy
        net.eval()
        print('Calculating validation loss...')
        for batch in val_generator:
            inpt, labels = batch[0].to(cuda), batch[1].to(cuda)
            output = net(inpt)
            val_loss += float(criterion(output, labels))
            divisor += 1
        print('Epoch: {} - Validation loss: {}'.format(EPOCH_NUM, val_loss/divisor))
        test_output = []
        for batch in te_generator:
            inpt = batch.to(cuda)
            output = net(inpt)
            test_output.append(output)
        test_output = np.array(test_output)
        np.savetxt('data/predictions.txt',test_output,fmt='%.6f')

    except FileNotFoundError:
        # This training script loads previous epoch if one exists and trains for one epoch before
        # saving. Structured this way to circumvent GPU memory issues on my not-so-powerful PC.
        if EPOCH_NUM != 1:
            net.load_state_dict(torch.load(PREV_MODEL_PATH))
        print('Initializing training script...')
        running_loss = 0
        for enum, batch in enumerate(tr_generator,1):
            optimizer.zero_grad()
            inpt, labels = batch[0].to(cuda), batch[1].to(cuda)
            output = net(inpt)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if enum%BATCHES_PER_REPORT == 0:
                print('Epoch: {} - Progress: {:.2f}% - MSE loss: {}'
                    .format(EPOCH_NUM, enum*BATCH_SIZE/val_idx*100,running_loss/BATCHES_PER_REPORT))
                running_loss = 0
        torch.save(net.state_dict(), MODEL_PATH)
    
    
    