import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np


# def rotate_matrix(input):
#     b, c, h, w = input.size()
#
#     features = input.view(b * c, h * w)
#
#     # features = input
#
#     return torch.cat(([features + torch.roll(features, i, 0) for i in range(0, len(features[0]))]), 0)
#
#
# class OverlayLoss(nn.Module):
#
#     def __init__(self):
#         super(OverlayLoss, self).__init__()
#
#     def forward(self, input, target):
#         O_input = rotate_matrix(input)
#         O_target = rotate_matrix(target)
#         return F.mse_loss((O_input, O_target))


# Equivalent optimization of the above code

class OverlayLoss(nn.Module):

    def __init__(self):
        super(OverlayLoss, self).__init__()

    def forward(self, input, target):

        b, c, h, w = input.size()

        i_features = input.view(b * c, h * w)

        t_features = target.view(b * c, h * w)

        return ((2 * b * c) * F.mse_loss(i_features, t_features) + 2 * F.mse_loss(torch.sum(i_features, 0), torch.sum(t_features, 0))).div(b * c)



if __name__ == "__main__":

    #testing

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device =  torch.device('cuda:1')


    x = Variable(torch.FloatTensor(np.random.random((1, 96, 128, 128))),requires_grad = True).to(device)

    img = Variable(torch.FloatTensor(np.random.random((1,96, 128, 128))), requires_grad=True).to(device)
    overlay_loss = OverlayLoss()
    loss = overlay_loss(x,img)
    loss.backward()

""" 
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
# res = rotate_matrix(torch.from_numpy(np.array(a)))
overlay_loss = OverlayLoss()
print(overlay_loss(torch.from_numpy(np.array(a)), torch.from_numpy(np.array(b)))) 

pass
"""

