# coding:utf-8
import json
import torch
import lib.utils.data as torchdata
import cv2
from torchvision import transforms
from scipy.misc import imread, imresize
import numpy as np
import pickle
import copy
import glob
import csv
import cv2
import time
import os
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
import matplotlib as mpl
from shapely.geometry import Polygon
import pdb

from torchvision import transforms
import torch
import pickle
import copy

import random
def rotate_im(image, angle,cxy=None):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    #
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW // 2) - cX
    M[1, 2] += (nH // 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))



def rotate_bound(shape, angle,cxy=None):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = shape
    #pdb.set_trace()
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    #
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW // 2) - cX
    M[1, 2] += (nH // 2) - cY
 
    # perform the actual rotation and return the image
    return M

def rotate(im_shape,text_polys,rd_rotate):

    
    m=rotate_bound(im_shape,rd_rotate)
    
    text_polys=text_polys.reshape([len(text_polys),-1,2])
    text_polys=np.concatenate((text_polys,np.ones([len(text_polys),text_polys.shape[1],1],dtype=np.float32)),-1)
    m=np.transpose(m,[1,0])
    text_polys=np.dot(text_polys,m.astype(np.float32))
    text_polys=text_polys.reshape(len(text_polys),-1)
    return text_polys



def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    p2=p2[np.newaxis,:].repeat(p3.shape[0],0)
    p1=p1[np.newaxis,:].repeat(p3.shape[0],0)

    return np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1,axis=-1)


def user_scattered_collate(batch):
    return batch

        

def round2nearest_multiple(x, p):
    nearest_x=int(max(p,np.round(x/p)*p))
    return nearest_x



class ValDataset(torchdata.Dataset):
    def __init__(self,ims=None,size=1800):


        root_im='/image_path/'

        ims=glob.glob(root_im+'*')

        self.im_names=ims
        self.img_size=round2nearest_multiple(int(size),32)
        print('img_size:{}'.format(self.img_size))
        
        self.size=len(self.im_names)
        print(self.size)
        self.img_transform = transforms.Compose([
                transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
                ])



    def __getitem__(self, index):
        img=cv2.imread(self.im_names[index])
        img_shape=img.shape
        scale_resize=self.img_size/max(img.shape[:2])
        w_resize=scale_resize*img.shape[1]
        h_resize=scale_resize*img.shape[0]


        w_resize=round2nearest_multiple(w_resize,32)
        h_resize=round2nearest_multiple(h_resize,32)
        img=cv2.resize(img,(w_resize,h_resize))
        img=self.img_transform(torch.from_numpy(img.astype(np.float32).transpose(2,0,1)))
        scale_h=float(h_resize)/img_shape[0]
        scale_w=float(w_resize)/img_shape[1]
        data_dict={}
        data_dict['im']=img.view(1,*img.size())
        data_dict['im_name']=os.path.basename(self.im_names[index])
        data_dict['scale_w']=scale_w
        data_dict['scale_h']=scale_h

        return data_dict


    def __len__(self):
        return self.size
        


