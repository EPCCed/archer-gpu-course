/*
 * This is a CUDA code that performs an iterative reverse edge 
 * detection algorithm.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2013 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/types.h>
#include <sys/time.h>

#include "reconstruct.h"

/* The actual CUDA kernel that runs on the GPU - 1D version by column */
__global__ void inverseEdgeDetect(float d_output[N+2][N+2], float d_input[N+2][N+2], \
					float d_edge[N+2][N+2])
{
  int imageCol, imageRow;
 
  /*
   * calculate global imageRow index for this thread  
   * from blockIdx.x, blockDim.x and threadIdx.x
   * remember to add 1 to account for halo    
   */

#ifdef SOLUTION 
  imageCol = blockIdx.x*blockDim.x + threadIdx.x + 1; 
#else 
  imageRow = blockIdx.x*blockDim.x + threadIdx.x + 1;
#endif 
  /*
   * loop over all columns of the image
   */
#ifdef SOLUTION 
  for (imageRow = 1; imageRow <= N; imageRow++) { 
#else 
  for (imageCol = 1; imageCol <= N; imageCol++) {
#endif 


      /* perform stencil operation */
      d_output[imageRow][imageCol] = (d_input[imageRow][imageCol-1] 
				      + d_input[imageRow][imageCol+1] 
				      + d_input[imageRow-1][imageCol] 
				      + d_input[imageRow+1][imageCol] 
				      - d_edge[imageRow][imageCol]) * 0.25; 

  }
}


  /* The actual CUDA kernel that runs on the GPU - 2D version */ 
  __global__ void inverseEdgeDetect2D(float d_output[N+2][N+2], float d_input[N+2][N+2], float d_edge[N+2][N+2]) 

{ 
  int imageCol, imageRow; 

  /* 
   * calculate global column index for this thread  
   * from blockIdx.x,blockDim.x and threadIdx.x    
   * remember to add 1 to account for halo     
   */ 
  imageCol = blockIdx.x*blockDim.x + threadIdx.x + 1; 

  /* 
   * calculate global imageRow index for this thread  
   * from blockIdx.y,blockDim.y and threadIdx.y 
   * remember to add 1 to account for halo    
   */ 
  imageRow = blockIdx.y*blockDim.y + threadIdx.y + 1; 

  /* perform stencil operation */  
  d_output[imageRow][imageCol] = (d_input[imageRow][imageCol-1]  
				  + d_input[imageRow][imageCol+1]  
				  + d_input[imageRow-1][imageCol]  
				  + d_input[imageRow+1][imageCol]  
				  - d_edge[imageRow][imageCol]) * 0.25;  


} 

