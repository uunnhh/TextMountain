#include <THC/THC.h>
#include "cuda/crop_and_resize_kernel.h"

extern THCState *state;


void crop_and_resize_gpu_forward(
    THCudaIntTensor * points_ptr,
    THCudaIntTensor * next_ptr,           // [y1, x1, y2, x2]
    THCudaIntTensor * instance_ptr,    // range in [0, batch_size)
    THCudaIntTensor * prob_ptr, //THCudaIntTensor * box_index
    THCudaIntTensor * circle_ptr
) {
    const int batch_size = THCudaIntTensor_size(state, next_ptr, 0);
    //const int depth = THCudaTensor_size(state, image, 1);
    const int image_height = THCudaIntTensor_size(state, next_ptr, 2);
    const int image_width = THCudaIntTensor_size(state, next_ptr, 3);
    const int points_num=THCudaIntTensor_size(state, points_ptr,0);
    //const int num_boxes = THCudaTensor_size(state, boxes, 0);

    // init output space
    //THCudaTensor_resize4d(state, crops, num_boxes, depth, crop_height, crop_width);
    //THCudaTensor_zero(state, crops);

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeLaucher(
        THCudaIntTensor_data(state, points_ptr),
        THCudaIntTensor_data(state, next_ptr),
        THCudaIntTensor_data(state, instance_ptr),
        THCudaIntTensor_data(state, prob_ptr),
        THCudaIntTensor_data(state,circle_ptr),
         batch_size, image_height, image_width,points_num,
        stream
    );
}

