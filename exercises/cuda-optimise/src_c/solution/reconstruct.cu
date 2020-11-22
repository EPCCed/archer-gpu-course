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

/* Data buffer to read edge data into */
float edge[N][N];

/* Data buffer for the resulting final image */
float img[N][N];

/* input and output arrays  (with halos) to be used in the main calculation */
float input[N+2][N+2];
float output[N+2][N+2];

/* an extra output array to be used for validation of the result */
float validate_output[N+2][N+2];



int main(int argc, char *argv[])
{
  int x, y;
  int i;
  int errors;

  double start_time_inc_data, end_time_inc_data;
  double cpu_start_time, cpu_end_time;

  float *d_input, *d_output, *d_edge;

  size_t memSize = (N+2) * (N+2) * sizeof(float);

  /* Print device details */
  int deviceNum;
  cudaGetDevice(&deviceNum);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceNum);
  printf("  Device name: %s\n", prop.name);


  printf("Image size: %dx%d\n", N, N);
  printf("ITERATIONS: %d\n", ITERATIONS);

#ifdef SOLUTION 

#define THREADSPERBLOCK_X 16 
#define THREADSPERBLOCK_Y 16 
 if ( N%THREADSPERBLOCK_X != 0 || N%THREADSPERBLOCK_Y != 0 ){ 
    printf("Error: THREADSPERBLOCK must exactly divide N in each dimension\n"); 
    exit(1); 
 } 
#else 


#define THREADSPERBLOCK 256

if ( N%THREADSPERBLOCK != 0 ){
    printf("Error: THREADSPERBLOCK must exactly divide N\n");
    exit(1);
 }

#endif 



  /* allocate memory on device */
  cudaMalloc(&d_input, memSize);
  cudaMalloc(&d_output, memSize);
  cudaMalloc(&d_edge, memSize);

  /* read in edge data */
  char filename[] = "edge2048x2048.dat";
  datread(filename, (void *)edge, N, N);

  /* zero buffer so that halo is zeroed */
  for (y = 0; y < N+2; y++) {
    for (x = 0; x < N+2; x++) {
      input[y][x] = 0.0;
    }
  }

  /* copy input to buffer with halo */
  for (y = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
       input[y+1][x+1] = edge[y][x];
    }
  }



  /* CUDA decomposition */
#ifdef SOLUTION 
    dim3 blocksPerGrid(N/THREADSPERBLOCK_X, N/THREADSPERBLOCK_Y,1); 
    dim3 threadsPerBlock(THREADSPERBLOCK_X, THREADSPERBLOCK_Y,1); 
#else 
    dim3 blocksPerGrid(N/THREADSPERBLOCK,1,1);
    dim3 threadsPerBlock(THREADSPERBLOCK,1,1);
#endif 

   printf("Blocks: %d %d %d\n",blocksPerGrid.x,blocksPerGrid.y,blocksPerGrid.z);
   printf("Threads per block: %d %d %d\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);


  /*
   * copy to all the GPU arrays. d_output doesn't need to have this data but
   * this will zero its halo
   */
  start_time_inc_data = get_current_time();
  cudaMemcpy( d_input, input, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_output, input, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_edge, input, memSize, cudaMemcpyHostToDevice);

  /* run on GPU */
  for (i = 0; i < ITERATIONS; i++) {

    /* run the kernel */
#ifdef SOLUTION 
    inverseEdgeDetect2D<<< blocksPerGrid, threadsPerBlock >>>((float (*)[N+2]) d_output, (float (*)[N+2]) d_input, (float (*)[N+2]) d_edge); 
#else 
    inverseEdgeDetect<<< blocksPerGrid, threadsPerBlock >>>((float (*)[N+2]) d_output, (float (*)[N+2]) d_input, (float (*)[N+2]) d_edge);
#endif 
 
    cudaDeviceSynchronize();

#ifdef SOLUTION 
    /* copy the output to the input on the device, */ 
    /*  ready for the next iteration */ 
    cudaMemcpy(d_input, d_output, memSize, cudaMemcpyDeviceToDevice); 

#else 
    /* copy the output data from device to host */
    cudaMemcpy(output, d_output, memSize, cudaMemcpyDeviceToHost);

    /* copy this same data from host to input buffer on device */
    /*  ready for the next iteration */ 
    cudaMemcpy( d_input, output, memSize, cudaMemcpyHostToDevice);
#endif 

  }

#ifdef SOLUTION 
  cudaMemcpy(output, d_output, memSize, cudaMemcpyDeviceToHost); 
#endif 

  end_time_inc_data = get_current_time();

  checkCUDAError("Main loop");

  /*
   * run on host for comparison
   */


  cpu_start_time = get_current_time();
  for (i = 0; i < ITERATIONS; i++) {

    /* perform stencil operation */
    for (y = 0; y < N; y++) {
      for (x = 0; x < N; x++) {
	validate_output[y+1][x+1] = (input[y+1][x] + input[y+1][x+2] +
				 input[y][x+1] + input[y+2][x+1] \
				 - edge[y][x]) * 0.25;
      }
    }
    
    /* copy output back to input buffer */
    for (y = 0; y < N; y++) {
      for (x = 0; x < N; x++) {
	input[y+1][x+1] = validate_output[y+1][x+1];
      }
    }
  }
  cpu_end_time = get_current_time();

/* Maximum difference allowed between host result and GPU result */
#define MAX_DIFF 0.01

  /* check that GPU result matches host result */
  errors = 0;
  for (y = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
      float diff = fabs(output[y+1][x+1] - validate_output[y+1][x+1]);
      if (diff >= MAX_DIFF) {
        errors++;
        //printf("Error at %d,%d (CPU=%f, GPU=%f)\n", x, y,	\
  	  //     validate_output[y+1][x+1],				\
		   //	      output[y+1][x+1]);
      }
    }
  }

  if (errors == 0) 
    printf("\n\n ***TEST PASSED SUCCESSFULLY*** \n\n\n");
  else
    printf("\n\n ***ERROR: TEST FAILED*** \n\n\n");

  /* copy result to output buffer */
  for (y = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
      img[y][x] = output[y+1][x+1];
    }
  }

  /* write PGM */
  char filename2[] = "output.pgm";
  pgmwrite(filename2, (void *)img, N, N);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_edge);

  printf("GPU Time (Including Data Transfer): %fs\n", \
	 end_time_inc_data - start_time_inc_data);
  printf("CPU Time                          : %fs\n", \
	 cpu_end_time - cpu_start_time);

  return 0;
}

