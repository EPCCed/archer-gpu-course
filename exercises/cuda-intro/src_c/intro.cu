/*
 * This is a simple CUDA code that negates an array of integers.
 * It introduces the concepts of device memory management, and
 * kernel invocation.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010-2022
 */

#include <stdio.h>
#include <stdlib.h>

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char*);

/* The number of integer elements in the array */
#define ARRAY_SIZE 256

/*
 * The number of CUDA blocks and threads per block to use.
 * These should always multiply to give the array size.
 * For the single block kernel, NUM_BLOCKS should be 1 and
 * THREADS_PER_BLOCK should be the array size
 */

#define NUM_BLOCKS  1
#define THREADS_PER_BLOCK 256

/* The actual kernel (basic single block version) */
/* Multiply all elements of the vector d_x by scalar a */

__global__ void scale_vector_1block(double a, double * d_x) {

  /* Part 2B: implement the operation for elements of d_x */

}

/* Multi-block version of kernel for part 2C */

__global__ void scale_vector(double a, double * d_x) {

  /* Part 2C: ... use more than one block ... */

}

/* Main routine */

int main(int argc, char *argv[]) {

  size_t sz = ARRAY_SIZE * sizeof(double);

  double * h_x = NULL;
  double * d_x = NULL;
  double * h_out = NULL;

  /* Print device details */

  int deviceNum;
  cudaGetDevice(&deviceNum);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceNum);
  printf("  Device name: %s\n", prop.name);

  /*
   * allocate memory on host
   * h_x holds the input array, h_out holds the result
   */

  h_x   = (double *) malloc(sz);
  h_out = (double *) malloc(sz);

  /*
   * allocate memory on device
   */
  /* Part 1A: allocate device memory */


  /* initialise host arrays */

  for (int i = 0; i < ARRAY_SIZE; i++) {
    h_x[i] = 1.0*i;
    h_out[i] = 0;
  }

  /* copy input array from host to GPU */
  /* Part 1B: copy host array h_x to device array d_x */


  /* run the kernel on the GPU */
  /* Part 2A: configure and launch kernel (un-comment and complete) */
  /* dim3 blocksPerGrid( ); */
  /* dim3 threadsPerBlock( ); */
  /* scale_vector_1block<<< , >>>( ); */


  /* wait for all threads to complete and check for errors */

  cudaDeviceSynchronize();
  checkCUDAError("kernel invocation");

  /* copy the result array back to the host */
  /* Part 1C: copy device array d_x to host array h_out */

  checkCUDAError("memcpy");

  /* print out the result */
  printf("Results: ");
  for (int i = 0; i < ARRAY_SIZE; i++) {
    printf("%d, ", h_out[i]);
  }
  printf("\n\n");

  /* free device buffer */
  /* Part 1D: free d_x */

  /* free host buffers */
  free(h_x);
  free(h_out);

  return 0;
}

/* Utility function to check for and report CUDA errors */

void checkCUDAError(const char * msg) {

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
