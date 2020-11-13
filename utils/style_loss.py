import torch
import torch.nn.functional as F
from torch import nn


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b , c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product
    #features_t = features.permute(1,0)
    #f = torch.matmul(theta_x, phi_bf)
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps. (.div(a * b * c * d))
    return G.div(a * b)


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()


    def forward(self, input,target):
        G_input = gram_matrix(input)
        G_target = gram_matrix(target)
        return  F.mse_loss(G_input, G_target)

