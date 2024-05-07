import torch
import torch.nn as nn
from torch.nn import functional as F

def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class Crnn(nn.Module):
    def __init__(self, num_freq, class_num):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################
        self.num_freq = num_freq
        self.class_num = class_num
        self.fc = nn.Linear(num_freq, class_num)
        self.batchnorm = nn.BatchNorm1d(num_freq)
        self.conv_block = nn.ModuleList()
        
        self.chns = [1, 16, 32, 64, 128]
        
        for i in range(len(self.chns)-1):
            self.conv_block.append(nn.Sequential(
                nn.Conv2d(self.chns[i], self.chns[i+1] , 3, padding=1),
                nn.BatchNorm2d(self.chns[i+1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
                nn.MaxPool2d(2)
            ))
        
        self.biGRU = nn.GRU(64*8, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(64*2, class_num)

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        bs, ts, nf = x.shape
        x_bn = self.batchnorm(x.permute(0,2,1)).view(bs,1,nf,ts)  # (bs, 1, 64, 501)
        for i in range(len(self.chns)-1):
            x_bn = self.conv_block[i](x_bn)
        timestep = x_bn.shape[-1]
        f, _ = self.biGRU(x_bn.view(bs,-1,timestep).permute(0,2,1))  # (bs, 62, 128)
        y = self.fc(f)  # (bs, 62, 10)
        out = torch.sigmoid(y)  # (bs, 62, 10)
        out_ = F.interpolate(out.permute(0,2,1), ts).permute(0,2,1)
        return out_
        

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, class_num)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, class_num)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }
