#include <math.h>
#include <stdio.h>
#include "crop_and_resize_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
     i += blockDim.x * gridDim.x)


__global__
void CropAndResizeKernel(
    const int nthreads, const int *points_ptr,
    const int *next_ptr, int *instance_ptr,const int *prob_ptr,int *circle_ptr,int batch, int image_height,
    int image_width,int points_num)
{
    CUDA_1D_KERNEL_LOOP(out_idx, nthreads)
    {
        
        int idx = out_idx;
        int x=points_ptr[idx*2];
        int y=points_ptr[idx*2+1];
     
        int b_in=0;

        int depth=2;
        int max_neg=1;
        int neg_now=0;
        int next_x;
        int next_y;
        int instance_idx;
        
        int next_xx;
        int next_yy;
        int num_search=1;
        
        next_x=next_ptr[(b_in * 2 + 0) * image_height * image_width+y*image_width+x];
        next_y=next_ptr[(b_in * 2 + 1) * image_height * image_width+y*image_width+x];
             
        instance_idx=instance_ptr[(b_in *1 + 0) * image_height * image_width+next_y*image_width+next_x];
        while(instance_idx==0&&neg_now<max_neg){
            if (num_search>(points_num+3)){
                circle_ptr[(b_in *1 + 0) * image_height * image_width+y*image_width+x]=0;
                circle_ptr[(b_in *1 + 0) * image_height * image_width+next_y*image_width+next_x]=0;


                break;
                
            }
            

            num_search=num_search+1;
            
            
            if (circle_ptr[(b_in *1 + 0) * image_height * image_width+next_y*image_width+next_x]==0){
                
                
                circle_ptr[(b_in *1 + 0) * image_height * image_width+y*image_width+x]=0;
                break;
            }
            if (next_x==x && next_y==y){
                circle_ptr[(b_in *1 + 0) * image_height * image_width+y*image_width+x]=0;
                
                break;
            }
            
            if (prob_ptr[(b_in *1 + 0) * image_height * image_width+next_y*image_width+next_x]==0){
                
                circle_ptr[(b_in *1 + 0) * image_height * image_width+y*image_width+x]=0;
                break;
                
            }




            
            next_xx=next_x;
            next_yy=next_y;
            next_x=next_ptr[(b_in * 2 + 0) * image_height * image_width+next_yy*image_width+next_xx];
            next_y=next_ptr[(b_in * 2 + 1) * image_height * image_width+next_yy*image_width+next_xx];

            
            
            
            instance_idx=instance_ptr[(b_in *1 + 0) * image_height * image_width+next_y*image_width+next_x];

            
        }
        
        if (instance_idx>0){
            instance_ptr[(b_in *1 + 0) * image_height * image_width+y*image_width+x]=instance_idx;
        }
       
  


    }
}


void CropAndResizeLaucher(
    const int *points_ptr,
    const int *next_ptr, int *instance_ptr,const int *prob_ptr,int *circle_ptr,int batch, int image_height,
    int image_width,int points_num, cudaStream_t stream)
{
    const int total_count = points_num;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count,points_ptr,
            next_ptr, instance_ptr,prob_ptr,circle_ptr,batch, image_height,
            image_width,points_num);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}

