#include <math.h>
#include <stdio.h>
#include "crop_and_resize_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
     i += blockDim.x * gridDim.x)
     
__global__
void CropAndResizeKernel(
    const int nthreads, const float *image_ptr, float *group_sum,float *groupNumsum,const int *boxID_ptr,int batch, int image_height,
    int image_width, int depth)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        
        int idx = out_idx;
        int boxID=boxID_ptr[idx];
        if (boxID!=0)
        {
        atomicAdd(
            groupNumsum+boxID,
            float(1.0)
        );
        float score_point=image_ptr[idx];
        atomicAdd(
            group_sum+boxID,
            score_point
        );}

    }
}



__global__
void CropAndResizeBackpropImageKernel(
    const int nthreads, const float *grads_ptr, int batch, int image_height,
    int image_width, int depth,
    float *grads_image_ptr,const int *boxID_ptr)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {

        int idx = out_idx;
        int boxID=boxID_ptr[idx];
        if (boxID!=0)
        {
            float grad_point=grads_ptr[boxID];
        atomicAdd(
            grads_image_ptr+idx,
            grad_point
        );
        }


    }
}


void CropAndResizeLaucher(
    const float *image_ptr, int batch, int image_height,
    int image_width, int depth, float *group_sum,float *groupNumsum,const int *boxID_ptr, cudaStream_t stream)
{
    const int total_count = batch * image_height * image_width * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count,image_ptr,group_sum,groupNumsum,boxID_ptr,batch, image_height, image_width,
            depth);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}


void CropAndResizeBackpropImageLaucher(
    const float *grads_ptr,  int batch, int image_height,
    int image_width, int depth,const int *boxID_ptr,
    float *grads_image_ptr, cudaStream_t stream)
{
    const int total_count = batch * image_height * image_width * depth;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeBackpropImageKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count, grads_ptr,  batch, image_height, image_width,
            depth, grads_image_ptr,boxID_ptr);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}
