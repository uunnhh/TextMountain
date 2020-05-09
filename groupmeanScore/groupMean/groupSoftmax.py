import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from ._ext import crop_and_resize as _backend
import pdb

class CropAndResizeFunction(Function):

    def __init__(self):
        pass
    def forward(self, image, boxID_ptr):
        #.new_zeros(num_rois, num_channels, out_h, out_w)
        #crops = torch.zeros_like(image)
        #pdb.set_trace()
        
        group_sum = image.new(int(boxID_ptr.max().cpu())+1).zero_()
        groupNumsum=image.new(int(boxID_ptr.max().cpu())+1).zero_()

        if image.is_cuda:
            _backend.crop_and_resize_gpu_forward(
                image,group_sum,groupNumsum,boxID_ptr)
        self.im_size=image.size()
        self.save_for_backward(boxID_ptr)

        return group_sum,groupNumsum

    def backward(self, grad_outputs,grad_outputs_sum):
        #print(grad_outputs)
        #print(grad_outputs_sum)
        boxID_ptr, = self.saved_tensors
        grad_outputs = grad_outputs.contiguous()
        #pdb.set_trace()
        grad_image = grad_outputs.new(*self.im_size).zero_()#.resize_(*self.im_size)
        if grad_outputs.is_cuda:
            _backend.crop_and_resize_gpu_backward(
                grad_outputs, boxID_ptr, grad_image
            )
        else:
            _backend.crop_and_resize_backward(
                grad_outputs, boxes, box_ind, grad_image
            )
        #print(grad_image)
        return grad_image, None


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self):
        super(CropAndResize, self).__init__()

    def forward(self, image, boxID_ptr):
        return CropAndResizeFunction()(image, boxID_ptr)
