#ifndef _CropAndResize_Kernel
#define _CropAndResize_Kernel

#ifdef __cplusplus
extern "C" {
#endif

void CropAndResizeLaucher(
    const float *image_ptr, int batch, int image_height,
    int image_width, int depth, float *group_sum,float *groupNumsum,const int *boxID_ptr, cudaStream_t stream);

void CropAndResizeBackpropImageLaucher(
    const float *grads_ptr,  int batch, int image_height,
    int image_width, int depth,const int *boxID_ptr,
    float *grads_image_ptr, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif