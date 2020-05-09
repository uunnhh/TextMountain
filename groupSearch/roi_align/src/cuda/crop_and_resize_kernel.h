#ifndef _CropAndResize_Kernel
#define _CropAndResize_Kernel

#ifdef __cplusplus
extern "C" {
#endif

void CropAndResizeLaucher(
    const int *points_ptr,
    const int *next_ptr,int *instance_ptr,const int *prob_ptr,int *circle_ptr,int batch, int image_height,
    int image_width,int points_num, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif