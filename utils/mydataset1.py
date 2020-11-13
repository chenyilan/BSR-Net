# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       dataset
   Author :         Hengrong LAN
   Date:            2018/12/26
-------------------------------------------------
   Change Activity:
                   2018/12/26:
-------------------------------------------------
"""

import numpy as np
import torch
import scipy
import os
import os.path
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import scipy.io as scio


def np_range_norm(image, maxminnormal=True, range1=True):

    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
        if maxminnormal:
            _min = image.min()
            _range = image.max() - image.min()
            narmal_image = (image - _min) / _range
            if range1:
               narmal_image = (narmal_image - 0.5) * 2
        else:
            _mean = image.mean()
            _std = image.std()
            narmal_image = (image - _mean) / _std

    return narmal_image



class ReconDataset(data.Dataset):
    __inputdata = []
    __outputimg = []
    __outputdata = []

    def __init__(self,root, train=True,transform=None):
        self.__inputdata = []
        self.__outputdata = []
        self.__outputimg = []

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        if train:
            folder = root + "Train/"
        else:
            folder = root + "Test/"
            
        
        
        for file in os.listdir(folder):
            #print(file)
            matdata = scio.loadmat(folder + file)
            prt = matdata['prt']

            nviews, _, _ =  prt.shape
            for index in range(0,nviews):
                prt[index,:,:]=np_range_norm(prt[index,:,:], maxminnormal=False, range1=False)
            out_data =  np.concatenate([prt[0:48,:,:],prt[80:,:,:]])
            pnum = np.sum(prt,axis=0)
            self.__inputdata.append(prt[48:80,:,:])

            self.__outputdata.append(out_data)
            
            self.__outputimg.append(pnum[np.newaxis,:,:])




        
            


    def __getitem__(self, index):

        rawdata =  self.__inputdata[index] 
        out_rawdata =self.__outputdata[index] #.reshape((1,1,2560,120))

        DAS = self.__outputimg[index]
           

        rawdata = torch.Tensor(rawdata)
        out_rawdata = torch.Tensor(out_rawdata)
        DAS = torch.Tensor(DAS)

        return rawdata, out_rawdata, DAS

    def __len__(self):
        return len(self.__inputdata)





if __name__ == "__main__":
    dataset_pathr = 'D:/detector_synthesis/brain/'

    mydataset = ReconDataset(dataset_pathr,train=False)
    #print(mydataset.__getitem__(3))
    train_loader = DataLoader(
        mydataset,
        batch_size=1, shuffle=True)
    batch_idx, (rawdata, out_rawdata,DAS) = list(enumerate(train_loader))[0]
    print(rawdata.size())
    print(out_rawdata.size())
    print(rawdata.max())
    print(rawdata.min())
    print(mydataset.__len__())






