void crop_and_resize_gpu_forward(
    THCudaIntTensor * points_ptr,
    THCudaIntTensor * next_ptr,           // [y1, x1, y2, x2]
    THCudaIntTensor * instance_ptr,    // range in [0, batch_size)
    THCudaIntTensor * prob_ptr,
    THCudaIntTensor * circle_ptr
);

