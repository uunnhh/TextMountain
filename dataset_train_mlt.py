# coding:utf-8
import os
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
import ssd_crop

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
def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """
    
    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))
    
    point = [
                [int(points[0]) , int(points[1])],
                [int(points[2]) , int(points[3])],
                [int(points[4]) , int(points[5])],
                [int(points[6]) , int(points[7])]
            ]
    edge = [
                ( point[1][0] - point[0][0])*( point[1][1] + point[0][1]),
                ( point[2][0] - point[1][0])*( point[2][1] + point[1][1]),
                ( point[3][0] - point[2][0])*( point[3][1] + point[2][1]),
                ( point[0][0] - point[3][0])*( point[0][1] + point[3][1])
    ]
    
    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    if summatory>0:
        print('wrong_clock')
        return 0
    else:
        return 1
        #raise Exception("Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards.")




def check_clock(polys):
    
    for poly in polys:
        
        if not validate_clockwise_points(poly.copy().reshape(-1)):
            poly[...]=poly[::-1]
            print('not_clock')
            if not validate_clockwise_points(poly.copy().reshape(-1)):
                print('wrong')

        
def generate_rbox(im_size, polys, tags,idx):
    check_clock(polys)

    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w,3), dtype=np.float32)
    template=np.zeros((h, w), dtype=np.uint8)
    training_mask = np.ones((h, w), dtype=np.uint8)
    
    
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < 10:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            continue
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            continue

        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        score_map_template=template.copy()
        cv2.fillPoly(score_map_template, poly[np.newaxis, :, :].astype(np.int32) , 1) 
        xy_inpoly=np.where(score_map_template==1)
        xy_inpoly=np.concatenate((xy_inpoly[1][:,np.newaxis],xy_inpoly[0][:,np.newaxis]),axis=-1)

        
        dis_0=point_dist_to_line(poly[0],poly[1],xy_inpoly)
        dis_1=point_dist_to_line(poly[1],poly[2],xy_inpoly)
        dis_2=point_dist_to_line(poly[2],poly[3],xy_inpoly)
        dis_3=point_dist_to_line(poly[3],poly[0],xy_inpoly)
        dis_02=dis_0+dis_2
        dis_13=dis_1+dis_3
        dis_cat=np.concatenate((dis_0[np.newaxis],dis_1[np.newaxis],dis_2[np.newaxis],dis_3[np.newaxis]),axis=0)

        dis_min=dis_cat.min(0)
        
        dis_mintwo=np.minimum(dis_13,dis_02)/2.
        

        dis_min_scale=dis_min/dis_mintwo
        geo_map[xy_inpoly[:,1],xy_inpoly[:,0],0]=dis_min_scale
        if not (dis_min_scale<=1).all():
            print('wrong')
            pdb.set_trace()


        def angle_line(point0,point1):
            line=point1-point0
            angle=np.angle(line[0]+line[1]*1j)
            return angle
        angle_0=angle_line(poly[0],poly[1])
        angle_1=angle_line(poly[1],poly[2])
        angle_2=angle_line(poly[2],poly[3])
        angle_3=angle_line(poly[3],poly[0])
        angle_cat=np.concatenate((angle_0[np.newaxis],angle_1[np.newaxis],angle_2[np.newaxis],angle_3[np.newaxis]),axis=0)+np.pi/2.
        weight_angle=np.maximum(dis_mintwo-dis_cat,0)

        weight_angle_sum=np.maximum(weight_angle.sum(0),1e-10)
        cos_point=(weight_angle*np.cos(angle_cat[:,np.newaxis])).sum(0)/weight_angle_sum
        sin_point=(weight_angle*np.sin(angle_cat[:,np.newaxis])).sum(0)/weight_angle_sum

        length=np.maximum((cos_point**2+sin_point**2)**0.5,1e-10)

        geo_map[xy_inpoly[:,1],xy_inpoly[:,0],1]=cos_point/length
        geo_map[xy_inpoly[:,1],xy_inpoly[:,0],2]=sin_point/length



        score_map[xy_inpoly[:,1],xy_inpoly[:,0]]=1


    score_map=np.concatenate((score_map[:,:,np.newaxis].astype(np.float32),geo_map[:,:]),-1)


    return score_map, geo_map, training_mask

# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p

def area_sort(roidb):
    for entry  in roidb:
        box=entry['segms']
        area=np.array([Polygon(var).area for var in box])
        area_argsort=(-area).argsort()
        entry['segms']=entry['segms'][area_argsort]
        entry['text_is_hard']=entry['text_is_hard'][area_argsort]


class TrainDataset(torchdata.Dataset):
    def __init__(self,batch_per_gpu):
        with open('./roidb_mlt.pik','rb') as f:
            roidb=pickle.load(f)
        root='/images_path/'
        for var in roidb:
            var['im_name']=root+os.path.basename(var['im_name'])
        
        self.img_transform = transforms.Compose([
                transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
                ])
        
        self.roidb=roidb
        

        area_sort(self.roidb)
        
        self.cur_idx = 0



        self.if_shuffled = False


        self.num_sample = len(self.roidb)
        self.crop_height=512
        
        self.batch_per_gpu=batch_per_gpu
        self.shrink_scale=1

        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def _get_sub_batch(self):
        batch_records=[]
        while True:
            # get a sample record
            this_sample = self.roidb[self.cur_idx]
            batch_records.append(this_sample)

            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.roidb)
            if len(batch_records)==self.batch_per_gpu:
                break
            
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.roidb)
            print('shuffle')
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()


        
        image_tensor=torch.zeros(self.batch_per_gpu, 3, self.crop_height, self.crop_height)
        score_tensor=torch.zeros(self.batch_per_gpu, 4, self.crop_height//self.shrink_scale, self.crop_height//self.shrink_scale)
        train_mask_tensor=torch.zeros(self.batch_per_gpu, 1, self.crop_height//self.shrink_scale, self.crop_height//self.shrink_scale)
        for batch_now,entry in enumerate(batch_records):     
            
            while(1):
                try:
                    im_fn = entry['im_name']
                    
                    im = cv2.imread(im_fn)
                    #pdb.set_trace()
                    text_polys=entry['segms'].copy()
                    text_tags=entry['text_is_hard'].copy()

                    
                    im_shape=im.shape[:2]
                    angle_random=0
                    if np.random.rand()<0.5:
                        
                        angle_random+=np.random.choice([90,0,180,270])
                        
                    angle_random+= np.random.uniform(-10,10)
                    if angle_random!=0:
                        text_polys=rotate(im_shape,text_polys,angle_random)
                        text_polys=text_polys.reshape([-1,4,2])
                        im=rotate_im(im, angle_random)
                    
                    background_ratio=-1
                    

                    im, text_polys, text_tags = ssd_crop.ssd_crop(im, text_polys, text_tags,self.crop_height)
                    
                   
                    new_h, new_w, _ = im.shape

                    score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags,batch_now)
                    
                    im=im.astype(np.float32,copy=False).transpose(2,0,1).copy()
                    image_tensor[batch_now]=self.img_transform(torch.from_numpy(im))
                    train_mask_tensor[batch_now]=torch.from_numpy(training_mask[::self.shrink_scale, ::self.shrink_scale, np.newaxis].astype(np.float32,copy=False).transpose(2,0,1))
                    
                    score_tensor[batch_now]=torch.from_numpy(score_map[::self.shrink_scale, ::self.shrink_scale].astype(np.float32,copy=False).transpose(2,0,1))
                    
                    break
                except Exception as e:
                    print(e)
                    continue
        batch_data={}
        batch_data['images']=image_tensor
        batch_data['score_maps']=score_tensor
        batch_data['training_masks']=train_mask_tensor
        return batch_data


    def __len__(self):
        return int(1e7) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass

