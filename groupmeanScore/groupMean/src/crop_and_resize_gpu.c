#include <THC/THC.h>
#include "cuda/crop_and_resize_kernel.h"

extern THCState *state;


void crop_and_resize_gpu_forward(
    THCudaTensor * image,
    THCudaTensor * group_sum,         // [y1, x1, y2, x2]
    THCudaTensor * groupNumsum, 

    THCudaIntTensor * boxID_ptr   // range in [0, batch_size)

) {
    const int batch_size = THCudaTensor_size(state, image, 0);
    const int depth = THCudaTensor_size(state, image, 1);
    const int image_height = THCudaTensor_size(state, image, 2);
    const int image_width = THCudaTensor_size(state, image, 3);

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeLaucher(
        THCudaTensor_data(state, image),
        batch_size, image_height, image_width,
        depth,
        THCudaTensor_data(state, group_sum),
        
        THCudaTensor_data(state, groupNumsum),
        THCudaIntTensor_data(state, boxID_ptr),
        stream
    );
}






void crop_and_resize_gpu_backward(
    THCudaTensor * grads,
    
    THCudaIntTensor * boxID_ptr,    // range in [0, batch_size)
    THCudaTensor * grads_image // resize to [bsize, c, hc, wc]
) {
    // shape
    const int batch_size = THCudaTensor_size(state, grads_image, 0);
    const int depth = THCudaTensor_size(state, grads_image, 1);
    const int image_height = THCudaTensor_size(state, grads_image, 2);
    const int image_width = THCudaTensor_size(state, grads_image, 3);



    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeBackpropImageLaucher(
        THCudaTensor_data(state, grads),
        batch_size, image_height, image_width,
        depth,
        THCudaIntTensor_data(state, boxID_ptr),
        THCudaTensor_data(state,grads_image),
        stream
    );
}
