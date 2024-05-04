import torch
import torch.nn as nn


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class Crnn(nn.Module):
    def __init__(self, num_freq, num_class):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################
        pass

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        ##############################
        pass

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, class_num)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, class_num)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }
