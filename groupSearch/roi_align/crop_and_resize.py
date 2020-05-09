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
        

    def forward(self,points_ptr, next_ptr, instance_ptr,prob_ptr):
        #pdb.set_trace()
        circle_ptr=torch.ones_like(prob_ptr)
        if 1:
            _backend.crop_and_resize_gpu_forward(
               points_ptr, next_ptr, instance_ptr,prob_ptr,circle_ptr)


        return None

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)

        if grad_outputs.is_cuda:
            _backend.crop_and_resize_gpu_backward(
                grad_outputs, boxes, box_ind, grad_image
            )
        else:
            _backend.crop_and_resize_backward(
                grad_outputs, boxes, box_ind, grad_image
            )

        return grad_image, None, None


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)
