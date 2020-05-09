import os
import torch
import torch.nn as nn
import torchvision
from . import resnet, resnext
from lib.nn import SynchronizedBatchNorm2d
import pdb
import sys
import pickle as pik
import copy
import cv2
import numpy as np
import numpy.random as npr
from simplification.cutil import simplify_coords
import shapely.geometry
from shapely.geometry import Polygon
import matplotlib as mpl
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as Patches

thres_center=0.6
score_thres_pixel=0.6

def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)



add_path('./groupSearch/')

from roi_align.crop_and_resize import CropAndResizeFunction as groupSearch

add_path('./groupmeanScore/')

from groupMean.groupSoftmax import CropAndResizeFunction as groupSoftmax




def image_idx_Tobox(image,valid,score_mean):

    idx_valid=np.where(valid)[0]+1

    bbox_lt=[]
    score_lt=[]
    for i in idx_valid:
        
        _, contours, _ = cv2.findContours((image==i).astype(np.uint8,copy=False),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_TREE RETR_EXTERNAL cv2.RETR_CCOMP
        try:
            contours=np.concatenate(contours,0)
        except:
            continue
        bbox_lt.append(cv2.boxPoints(cv2.minAreaRect(contours.reshape(-1,2))))
        score_lt.append(score_mean[i-1])
    return np.array(bbox_lt),np.array(score_lt)




def image_idx_Tobox_curve(image,valid):

    idx_valid=np.where(valid)[0]+1
    bbox_lt=[]
    for i in idx_valid:
        
        _, contours, _ = cv2.findContours((image==i).astype(np.uint8,copy=False),cv2.RETR_TREE    ,cv2.CHAIN_APPROX_SIMPLE)
        
        c = max(contours, key = cv2.contourArea).reshape(-1,2).astype(np.float32,copy=False)

        c=simplify_coords(c, 1.0)
        if c.shape[0]<3:
            continue
        Polygon_c=Polygon(c)

        if not Polygon_c.is_valid:

            Polygon_c=Polygon_c.buffer(0)
            if not Polygon_c.is_valid:
                pass
                
            if type(Polygon_c) is shapely.geometry.multipolygon.MultiPolygon:
                
                continue
            try:
                c=np.array(list(Polygon_c.exterior.coords),dtype=np.float32).reshape(-1,2)
            except:
                
                continue
                
           
        contours=[c]
        bbox_lt.append(contours)
        
    return bbox_lt




class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc



class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        if segSize is None: # training
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        else: # inference
            pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

def fully_toconv(in_planes=256, out_planes=1024, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, 512, kernel_size=7, stride=1,
                     padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1,
                     padding=0, bias=True),
            nn.ReLU(inplace=True),   
            nn.Conv2d(512, 8, kernel_size=1, stride=1,
                     padding=0, bias=True),         
            )


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    def build_encoder_textmountain(self,weights=''):
        pretrained = True if len(weights) == 0 else False

        orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
        net_encoder = Resnet(orig_resnet)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder



    def build_decoder_textmountain(self,weights=''):
        net_decoder=decode_textmountain()
        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50_dilated8':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet50_dilated16':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101_dilated8':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet101_dilated16':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(self, arch='ppm_bilinear_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        if arch == 'c1_bilinear_deepsup':
            net_decoder = C1BilinearDeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1_bilinear':
            net_decoder = C1Bilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear':
            net_decoder = PPMBilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear_deepsup':
            net_decoder = PPMBilinearDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        elif arch == 'upernet_tmp':
            net_decoder = UPerNetTmp(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]



class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


# last conv, bilinear upsample
class C1BilinearDeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1BilinearDeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv, bilinear upsample
class C1Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling, bilinear upsample
class PPMBilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, bilinear upsample
class PPMBilinearDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


class decode_textmountain(nn.Module):
    def __init__(self, num_class=5, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=256):
        super(decode_textmountain, self).__init__()
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)    
        self.conv_up=nn.Sequential(
                conv3x3_bn_relu(1024, 256, 1),
                nn.Conv2d(256, num_class, kernel_size=1)

            )
        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

   





    def forward(self, conv_out, segSize=None):
        #conv5 = conv_out[-1]


        
        f=self.fpn_in[-1](conv_out[-1])
        fpn_feature_list = [f]
        
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.upsample(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))
        
        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.upsample(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        
        fusion_out = torch.cat(fusion_list, 1)

        f=self.conv_up(fusion_out)
        
        f=nn.functional.upsample(
                f, size=(segSize[2],segSize[3]), mode='bilinear', align_corners=False)
        
        return f



# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )




    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.upsample(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.upsample(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x


  

def get_cc_eval_cuda(img,score,thres=0.7):

    img = img.copy()

    connectivity = 4


    output= cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)


    torch.cuda.synchronize()
    score_i=torch.from_numpy(output[1]).cuda()
    score_sum,score_num=groupSoftmax().forward( score[:,1:],score_i.unsqueeze(0).unsqueeze(0))
    score_mean=score_sum[1:]/torch.clamp(score_num[1:],min=1e-10)
    score_sum=score_sum.cpu().data.numpy()
    score_mean=score_mean.cpu().data.numpy()
    score_num=score_num.cpu().data.numpy()

    valid=(score_mean>thres)&(score_num[1:]>5.)

    torch.cuda.synchronize()

    return valid,score_i,output[1],score_mean




def OHNM_single_image(scores, n_pos, neg_mask):
    """Online Hard Negative Mining.
        scores: the scores of being predicted as negative cls
        n_pos: the number of positive samples 
        neg_mask: mask of negative samples
        Return:
            the mask of selected negative samples.
            if n_pos == 0, top 10000 negative samples will be selected.
    """
    def has_pos():
        return n_pos * 3
    def no_pos():
        return 10000
    if n_pos>0:
        n_neg=has_pos()
    else:
        n_neg=no_pos()

    max_neg_entries = int(torch.sum(neg_mask).cpu().data)
    
    n_neg = min(n_neg, max_neg_entries)

    if n_neg==0:
        return torch.zeros_like(scores)
    
    neg_conf=scores*neg_mask.type(torch.cuda.FloatTensor)
    values,idxs=torch.topk(-neg_conf,k=n_neg)
    threshold = values[-1]
    selected_neg_mask =(scores <= -threshold)&(neg_mask)
    return selected_neg_mask.type(torch.cuda.FloatTensor)

def OHNM_batch(neg_conf, pos_mask, neg_mask):
    selected_neg_mask = []
    for image_idx in range(neg_conf.size(0)):
        image_neg_conf = neg_conf[image_idx, :].view(-1)
        image_neg_mask = neg_mask[image_idx, :].view(-1)
        image_pos_mask = pos_mask[image_idx, :].view(-1)
        n_pos = int(torch.sum(image_pos_mask).cpu().data)
        selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))
        
    selected_neg_mask = torch.stack(selected_neg_mask)
    return selected_neg_mask



class Module_textmountain(nn.Module):
    def __init__(self, net_enc, net_dec, score_thres=0.7, ctw=0):
        super(Module_textmountain, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit =nn.CrossEntropyLoss(ignore_index=-1,size_average=False,reduce=False)
        
        self.score_thres=score_thres
        offset=1
        self.padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        self.cuda_group=1

        self.ctw=ctw

    def forward(self, feed_dict, *, segSize=None,is_Train=True,img_name=None,save_dir=None):

        if is_Train:
            
            pred = self.decoder(self.encoder(feed_dict['images'], return_feature_maps=True),feed_dict['images'].size())
            pred_0=pred[:,:2]
            pred_1=nn.functional.sigmoid(pred[:,2])
            pred_cossin=(nn.functional.sigmoid(pred[:,3:5])-0.5)*2

            target_0=feed_dict['score_maps'][:,0]
            target_1=feed_dict['score_maps'][:,1]
            if not (target_1<=1).all():
                print('wrong')
                
            target_cossin=feed_dict['score_maps'][:,2:]

            pred_0_softmax=nn.functional.softmax(pred[:,:2],dim=1)

            pos_map_0=((target_0==1)&(feed_dict['training_masks'][:,0]!=0))
            neg_map_0=((target_0==0)&(feed_dict['training_masks'][:,0]!=0))
            selected_neg_pixel_mask=OHNM_batch(-pred_0_softmax[:,1],pos_map_0,neg_map_0)
            selected_neg_pixel_mask=selected_neg_pixel_mask.view(target_0.size())
            selected_pixel_mask_0=pos_map_0.type(torch.cuda.FloatTensor)+selected_neg_pixel_mask

            valid_0=selected_pixel_mask_0*feed_dict['training_masks'][:,0]
            loss_0 = self.crit(pred_0, target_0.type(torch.cuda.LongTensor))

            
            loss_1=torch.abs(pred_1*5-target_1*5)

            loss_cossin=torch.abs(pred_cossin*2.5-target_cossin*2.5)
            valid_1=pos_map_0.type(torch.cuda.FloatTensor)*feed_dict['training_masks'][:,0]
            loss_0=(loss_0*valid_0).sum()/(valid_0.sum()+1e-10)
            loss_1=(loss_1*valid_1).sum()/(valid_1.sum()+1e-10)


            valid_cossin=valid_1*(pred_1<=thres_center).detach().type(torch.float32)
            loss_cossin=(loss_cossin*valid_cossin.unsqueeze(1)).sum()/(valid_cossin.sum()+1e-10)

            loss_dict={'loss_0':loss_0,'loss_1':loss_1,'loss_cossin':loss_cossin}#
            
            return sum(loss_dict.values()),loss_dict

                


        elif not self.ctw: # inference
           
            pred=self.decoder(self.encoder(feed_dict['images'], return_feature_maps=True),feed_dict['images'].size())
            
            pred_0=nn.functional.softmax(pred[:,:2],dim=1)
            pred_1=nn.functional.sigmoid(pred[:,2:3])*1.

            win_size=3

            padded_maps = self.padding(pred_1)

            _, indices  = nn.functional.max_pool2d(
                padded_maps,
                kernel_size = win_size, 
                stride = 1, 
                return_indices = True)
            
            batch_size, num_channels, h, w = pred_1.size()
            indices_2d=torch.zeros([2,indices.size(2),indices.size(3)],dtype=torch.int32,device=torch.device('cuda'))#.cuda()

            
            indices_2d[1,:,:]=indices[0,0]/(w+2)-1
            indices_2d[0,:,:]=indices[0,0]%(w+2)-1
            


            shrink_scores=(pred_1[0,0]>thres_center).cpu().data.numpy().astype(np.uint8,copy=False)
            valid,score_i,score_i_numpy,score_mean=get_cc_eval_cuda(shrink_scores,pred_0,thres=self.score_thres)

            
            
            points_ptr=torch.nonzero((pred_0[0,1]>score_thres_pixel)&(score_i==0))
            if len(points_ptr)==0:
                return np.zeros([0,4,2],dtype=np.float32),None
            
            points_ptr=points_ptr[:,torch.tensor([1,0])].int().contiguous()
            
            pred_0_thres=(pred_0[:,1]>score_thres_pixel).type(torch.int32)
           




            groupSearch().forward(points_ptr, indices_2d.unsqueeze(0), score_i.unsqueeze(0).unsqueeze(0),pred_0_thres.unsqueeze(0))
            

            score_border=score_i.cpu().data.numpy()
            bbox,score_box=image_idx_Tobox(score_border,valid,score_mean)


            return bbox,score_box


        else: # inference
           
            pred=self.decoder(self.encoder(feed_dict['images'], return_feature_maps=True),feed_dict['images'].size())

            pred_0=nn.functional.softmax(pred[:,:2],dim=1)
            
            pred_1=nn.functional.sigmoid(pred[:,2:3])*1.
            




            win_size=3
            offset = (win_size - 1) // 2
            padding = torch.nn.ConstantPad2d(offset, float('-inf'))
            
            padded_maps = padding(pred_1)
            
            _, indices  = nn.functional.max_pool2d(
                padded_maps,
                kernel_size = win_size, 
                stride = 1, 
                return_indices = True)



            batch_size, num_channels, h, w = pred_1.size()
            indices_2d=torch.zeros([2,indices.size(2),indices.size(3)],dtype=torch.int32,device=torch.device('cuda'))#.cuda()

            
            indices_2d[1,:,:]=indices[0,0]/(w+2)-1
            indices_2d[0,:,:]=indices[0,0]%(w+2)-1

            assert indices_2d[0,:,:].max()<=w-1
            assert indices_2d[1,:,:].max()<=h-1
            assert indices_2d[0,:,:].min()>=0
            assert indices_2d[1,:,:].min()>=0            
            
            
            shrink_scores_float=pred_1[0,0].cpu().data.numpy()

            pred_0_numpy=pred_0.cpu().data.numpy()
            shrink_scores=(shrink_scores_float>0.6).astype(np.uint8,copy=False)


            valid,score_i,score_i_numpy,score_mean=get_cc_eval_cuda(shrink_scores,pred_0)
        
            y_border,x_border=np.where((pred_0_numpy[0,1]>0.6)&(score_i_numpy==0))
            
            if y_border.shape[0]==0:
                return []

            points_ptr=torch.from_numpy(np.concatenate((x_border[:,np.newaxis],y_border[:,np.newaxis]),-1).astype(np.int32,copy=False)).cuda()
            
            pred_0_thres=(pred_0[:,1]>0.6).type(torch.int32)
            
            groupSearch().forward(points_ptr, indices_2d.unsqueeze(0), score_i.unsqueeze(0).unsqueeze(0),pred_0_thres.unsqueeze(0))
            score_border=score_i.cpu().data.numpy()
            
            
            bbox=image_idx_Tobox_curve(score_border,valid)
            
            
            

            
            bbox=[var[0] for var in bbox]
            return bbox



