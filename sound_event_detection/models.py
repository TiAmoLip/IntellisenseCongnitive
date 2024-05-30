import torch
import torch.nn as nn
from torch.nn import functional as F

def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class Crnn(nn.Module):
    def __init__(self, num_freq, num_class, use_gru=False, base_channels=16, ffn_hidden=64, num_channels=4):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################
        super().__init__()
        self.num_freq = num_freq
        self.num_class = num_class
        self.fc = nn.Linear(num_freq, num_class)
        self.batchnorm = nn.BatchNorm1d(num_freq)
        self.conv_block = nn.ModuleList()
        # self.meta_shape = 
        # self.chns = [1, 16, 32, 64, 128] # basechannels, num_channels
        self.chns = [1] + [base_channels*(2**i) for i in range(num_channels)]
        
        for i in range(len(self.chns)-1):
            self.conv_block.append(nn.Sequential(
                nn.Conv2d(self.chns[i], self.chns[i+1] , 3, padding=1),
                nn.BatchNorm2d(self.chns[i+1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
                nn.MaxPool2d(2)
            ))
        if use_gru:
            RNN = nn.GRU
        else:
            RNN = nn.LSTM
        meta = torch.randn((1, 501, 64))
        bs, ts, nf = meta.shape
        x_bn = self.batchnorm(meta.permute(0,2,1)).view(bs,1,nf,ts)
        for i in range(len(self.chns)-1):
            x_bn = self.conv_block[i](x_bn)
        
        
        self.biGRU = RNN(x_bn.shape[1]*x_bn.shape[2], ffn_hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(ffn_hidden*2, num_class) #ffn_hidden

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     prob: [batch_size, time_steps, num_class]
        ##############################
        bs, ts, nf = x.shape
        x_bn = self.batchnorm(x.permute(0,2,1)).view(bs,1,nf,ts)
        for i in range(len(self.chns)-1):
            x_bn = self.conv_block[i](x_bn)
        # print("after conv block: ", x_bn.shape)
        timestep = x_bn.shape[-1]
        f, _ = self.biGRU(x_bn.view(bs,-1,timestep).permute(0,2,1))
        y = self.fc(f)
        out = torch.sigmoid(y)
        out_ = F.interpolate(out.permute(0,2,1), ts).permute(0,2,1)
        return out_
        

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, num_class)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, num_class)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }
