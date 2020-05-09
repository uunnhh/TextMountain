import os
import time
import random
import argparse
from distutils.version import LooseVersion
import torch
import torch.nn as nn
from torchvision import transforms
from models import ModelBuilder, SegmentationModule,Module_textmountain
from utils import AverageMeter
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata
import pdb
import glob
from dataset_test import ValDataset,user_scattered_collate
import cv2
import numpy as np

save_dir='./save_path'
test_size=1800
os.system('mkdir %s'%save_dir)
os.system('rm %s*'%save_dir)


def main():
    
    builder = ModelBuilder()
    net_encoder=builder.build_encoder_textmountain
    net_decoder = builder.build_decoder_textmountain


    encoder_path='model_path'
    decoder_path='model_path'
    
    net_encoder = net_encoder(encoder_path)
    net_decoder = net_decoder(decoder_path)

    segmentation_module=Module_textmountain(
    net_encoder,net_decoder,score_thres=0.7)
    segmentation_module.cuda()
    segmentation_module.eval()
    dataset_train = ValDataset(size=test_size
       )
    loader_train = torchdata.DataLoader(
        dataset_train,
        batch_size=1,  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=int(1),
        drop_last=True,
        pin_memory=True)
    iterator_train = iter(loader_train)

    for i_data in range(len(iterator_train)):
        batch_data = next(iterator_train)
        with torch.no_grad():
            feed_dict={}           
            feed_dict['images']=batch_data[0]['im'].cuda()

            pred_polygon,score_box=segmentation_module.forward(feed_dict,is_Train=False,img_name=batch_data[0]['im_name'],save_dir=save_dir)
            
            
            if len(pred_polygon)==0:
                continue
            im=batch_data[0]['im_name']
            scale_w=batch_data[0]['scale_w']
            scale_h=batch_data[0]['scale_h']
            pred_polygon[:,:,0]/=scale_w
            pred_polygon[:,:,1]/=scale_h
            if pred_polygon is not None:
                res_file = os.path.join(
                    save_dir,
                    'res_{}.txt'.format(
                        os.path.basename(im)[3:].split('.')[0]))
                with open(res_file, 'w') as f:
                    for box,score_one in zip(pred_polygon,score_box):
                        box=box.astype(np.int32)
                        
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                            continue
                            
                        f.write('{},{},{},{},{},{},{},{},{}\r\n'.format(
                            box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],score_one
                        ))
                

main()
print('done')