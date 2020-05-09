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
import matplotlib as mpl
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon,LinearRing
import pdb
import ssd_crop
from torchvision import transforms
import torch
import pickle
import copy
import random
min_crop_side_ratio=0.1
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


def check_clock(polys):

    for poly in polys:

        if not LinearRing(poly).is_ccw:
            print(poly)
            poly[...]=poly[::-1]
            print('not_clock')
            if not LinearRing(poly):
                print('wrong')
                print(poly)            


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

def valid_disToline(p1,p2,p3):

    p2=p2[np.newaxis,:].repeat(p3.shape[0],0)
    p1=p1[np.newaxis,:].repeat(p3.shape[0],0)
    dis_line=np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1,axis=-1)
    dis_point1=np.linalg.norm(p3 - p1,axis=-1)

    dis_point2=np.linalg.norm(p3 - p2,axis=-1)
    line13=p3-p1
    line12=p2-p1

    p12x=(line13*line12).sum(1)

    p12x=p12x/np.linalg.norm(line12,axis=-1)**2


    dis=((p12x>0)&(p12x<1))*dis_line+(p12x<=0)*dis_point1+(p12x>=1)*dis_point2

    return dis


def valid_dis(p1,p2,p3,angle1,angle2):

    p2=p2[np.newaxis,:].repeat(p3.shape[0],0)
    p1=p1[np.newaxis,:].repeat(p3.shape[0],0)
    dis_line=np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1,axis=-1)
    dis_point1=np.linalg.norm(p3 - p1,axis=-1)

    dis_point2=np.linalg.norm(p3 - p2,axis=-1)
    line13=p3-p1
    line12=p2-p1

    p12x=(line13*line12).sum(1)

    p12x=p12x/np.linalg.norm(line12,axis=-1)**2


    dis=((p12x>0)&(p12x<1))*dis_line+(p12x<=0)*dis_point1+(p12x>=1)*dis_point2

    weight1=1-p12x[:].copy()
    weight2=p12x[:]

    angle_line=angle_weight_add(angle1[np.newaxis],angle2[np.newaxis],weight1,weight2)
    
    angle_min=(p12x<=0)*angle1+(p12x>=1)*angle2+((p12x>0)&(p12x<1))*angle_line
    
    return dis,angle_min


def angle_weight_add(angle0,angle1,weight1,weight2):
    radical0=np.cos(angle0)+np.sin(angle0)*1j
    radical1=np.cos(angle1)+np.sin(angle1)*1j
    radical1_0=radical1-radical0
    radicalnow=radical0+radical1_0*weight2/(weight1+weight2)
    return np.angle(radicalnow)








def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        if xmax - xmin < min_crop_side_ratio*w or ymax - ymin < min_crop_side_ratio*h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == polys.shape[1])[0]
            
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags
    print('wrong_max_iter')
    return im, polys, tags




def point_dist_to_curveLine(line,point):

    disToline=[]
    disToline_valid=[]
    dis_toPOINT_lt=[]
    angle_lines=[]
    length_lines=[]
    angle_lines=[]
    length_lines=[]
    angle_min_lt=[]
    for i in range(len(line)-1):
        angle_radical=line[i+1]-line[i]
        length_lines.append(np.linalg.norm(angle_radical))
        angle_lines.append(np.angle(angle_radical[0]+1j*angle_radical[1]))
    angle_lines=[angle_lines[0]]+angle_lines+[angle_lines[-1]]
    angle_lines=np.array(angle_lines)
    assert np.array(length_lines).min()>0
    
    angle_points=angle_weight_add(angle_lines[:-1],angle_lines[1:],0.5,0.5)

    
    for i in range(len(line)-1):
        
        dis,angle_min=valid_dis(line[i], line[i+1], point,angle_points[i],angle_points[i+1])
        disToline.append(dis[np.newaxis])
        angle_min_lt.append(angle_min[np.newaxis])
        
    angle_min_lt=np.concatenate((angle_min_lt),axis=0)
    disToline=np.concatenate((disToline),axis=0)
    
    dis_final=disToline.min(0)
    
    angle_final=angle_min_lt[disToline.argmin(0),np.arange(disToline.shape[1])]
    return angle_final,dis_final






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


        
def generate_rbox(im_size, polys, tags,idx,map_height,map_width):

    check_clock(polys)
    
    score_map = np.zeros((map_height,map_width), dtype=np.uint8)
    
    angle_border=np.zeros((map_height,map_width), dtype=np.float32)
    geo_map = np.zeros((map_height,map_width,3), dtype=np.float32)
    template=np.zeros((map_height,map_width), dtype=np.uint8)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((map_height,map_width), dtype=np.uint8)
    
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]
        
        score_map_template=template.copy()
        cv2.fillPoly(score_map_template, poly[np.newaxis, :, :].astype(np.int32) , 1) 
        xy_inpoly=np.where(score_map_template==1)
        xy_inpoly=np.concatenate((xy_inpoly[1][:,np.newaxis],xy_inpoly[0][:,np.newaxis]),axis=-1)
        angle_0,dis_0=point_dist_to_curveLine(poly[:7].copy(),xy_inpoly.copy())
        angle_2,dis_2=point_dist_to_curveLine(poly[7:].copy(),xy_inpoly.copy())

        
        dis_1=valid_disToline(poly[6],poly[7],xy_inpoly)
        dis_3=valid_disToline(poly[-1],poly[0],xy_inpoly)
        
        dis_02=dis_0+dis_2
        dis_13=dis_1+dis_3
        
        dis_cat=np.concatenate((dis_0[np.newaxis],dis_1[np.newaxis],dis_2[np.newaxis],dis_3[np.newaxis]),axis=0)

        dis_min=dis_cat.min(0)
        
        dis_minmax=dis_min.max()
        
        if dis_minmax < 5:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            continue
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            continue
        
        dis_mintwo=np.minimum(dis_13,dis_02)/2.

        
        dis_min_scale=dis_min/dis_mintwo
        
        geo_map[xy_inpoly[:,1],xy_inpoly[:,0],0]=dis_min_scale


        def angle_line(point0,point1):
            line=point1-point0
            angle=np.angle(line[0]+line[1]*1j)
            return angle
        
        angle_1=angle_line(poly[6],poly[7]).repeat(angle_0.shape[0])
        
        angle_3=angle_line(poly[-1],poly[0]).repeat(angle_0.shape[0])
        angle_cat=np.concatenate((angle_0[np.newaxis],angle_1[np.newaxis],angle_2[np.newaxis],angle_3[np.newaxis]),axis=0)+np.pi/2.
        weight_angle=np.maximum(dis_mintwo-dis_cat,0)
        
        weight_angle_sum=np.maximum(weight_angle.sum(0),1e-10)
        cos_point=(weight_angle*np.cos(angle_cat[:])).sum(0)/weight_angle_sum
        sin_point=(weight_angle*np.sin(angle_cat[:])).sum(0)/weight_angle_sum
        
        length=np.maximum((cos_point**2+sin_point**2)**0.5,1e-10)

        geo_map[xy_inpoly[:,1],xy_inpoly[:,0],1]=cos_point/length
        geo_map[xy_inpoly[:,1],xy_inpoly[:,0],2]=sin_point/length
        

        score_map[xy_inpoly[:,1],xy_inpoly[:,0]]=1
        

    score_map=np.concatenate((score_map[:,:,np.newaxis].astype(np.float32),geo_map[:,:]),-1)
    

    return score_map, geo_map, training_mask#,angle_border

# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p

class TrainDataset(torchdata.Dataset):
    def __init__(self,batch_per_gpu):
        
        with open('./roidb_ctw.pik','rb') as f:
            roidb=pickle.load(f)
        
        root='/image_path/'
        for var in roidb:
            var['im_name']=root+os.path.basename(var['im_name'])


        index = np.arange(0, len(roidb))

        
        self.img_transform = transforms.Compose([
                transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
                ])
        self.roidb=roidb

        # override dataset length when trainig with batch_per_gpu > 1
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
            
            im_fn = entry['im_name']
            im = cv2.imread(im_fn)
            text_polys=entry['segms'].copy()
            text_tags=entry['text_is_hard'].copy()
            

            im_shape=im.shape[:2]
            angle_random=0
            if np.random.rand()<0.5:
                
                angle_random+=np.random.choice([90,0,180,270])
                
            angle_random+= np.random.uniform(-10,10)
            if angle_random!=0:
                text_polys=rotate(im_shape,text_polys,angle_random)
                text_polys=text_polys.reshape([-1,14,2])
                im=rotate_im(im, angle_random)


            rd_scale = np.random.choice(np.array([0.5, 1, 2.0, 3.0]))

            im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
            text_polys *= rd_scale
            
           

            im_shape=im.shape[:2]

            text_polys[:, :, 0] = np.clip(text_polys[:, :, 0], 0, im_shape[1]-1)
            text_polys[:, :, 1] = np.clip(text_polys[:, :, 1], 0, im_shape[0]-1)

            
            background_ratio=1./8
            if np.random.rand() < background_ratio:
                
                im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                if text_polys.shape[0] > 0:
                    # cannot find background
                    continue
                # pad and resize image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, self.crop_height])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = cv2.resize(im_padded, dsize=(self.crop_height, self.crop_height))
                score_map = np.zeros((self.crop_height, self.crop_height,4), dtype=np.float32)
                
                geo_map = np.zeros((self.crop_height, self.crop_height, 3), dtype=np.float32)
                training_mask = np.ones((self.crop_height, self.crop_height), dtype=np.float32)

            else:
                im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                if text_polys.shape[0] == 0:
                    print('foreground_false')
                    continue
                h, w, _ = im.shape

                # pad the image to the training input size or the longer side of image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, self.crop_height])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = self.crop_height
                resize_w = self.crop_height
                im = cv2.resize(im, dsize=(resize_w, resize_h))
                resize_ratio_3_x = resize_w/float(new_w)
                resize_ratio_3_y = resize_h/float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
                score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags,batch_now,new_h,new_w)


           
           
            image_tensor[batch_now]=self.img_transform(torch.from_numpy(im.astype(np.float32,copy=False).transpose(2,0,1)))
            train_mask_tensor[batch_now]=torch.from_numpy(training_mask[::self.shrink_scale, ::self.shrink_scale, np.newaxis].astype(np.float32,copy=False).transpose(2,0,1))
            
            score_tensor[batch_now]=torch.from_numpy(score_map[::self.shrink_scale, ::self.shrink_scale].astype(np.float32,copy=False).transpose(2,0,1))

        batch_data={}
        batch_data['images']=image_tensor
        batch_data['score_maps']=score_tensor
        batch_data['training_masks']=train_mask_tensor
        return batch_data


    def __len__(self):
        return int(1e7) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass

