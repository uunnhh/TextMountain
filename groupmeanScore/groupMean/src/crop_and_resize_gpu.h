void crop_and_resize_gpu_forward(
    THCudaTensor * image,
    THCudaTensor * group_sum,         // [y1, x1, y2, x2]
    THCudaTensor * groupNumsum, 

    THCudaIntTensor * boxID_ptr
);

void crop_and_resize_gpu_backward(
    THCudaTensor * grads,
    
    THCudaIntTensor * boxID_ptr,    // range in [0, batch_size)
    THCudaTensor * grads_image // resize to [bsize, c, hc, wc]
);