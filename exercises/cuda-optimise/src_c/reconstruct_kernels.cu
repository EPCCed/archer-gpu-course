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

  imageRow = blockIdx.x*blockDim.x + threadIdx.x + 1;
  /*
   * loop over all columns of the image
   */
  for (imageCol = 1; imageCol <= N; imageCol++) {


      /* perform stencil operation */
      d_output[imageRow][imageCol] = (d_input[imageRow][imageCol-1] 
				      + d_input[imageRow][imageCol+1] 
				      + d_input[imageRow-1][imageCol] 
				      + d_input[imageRow+1][imageCol] 
				      - d_edge[imageRow][imageCol]) * 0.25; 

  }
}









