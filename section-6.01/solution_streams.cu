/*
 * Streams exercise
 *
 * An implementation of the blas level 2 routine dger(), which is
 * the operation
 *
 *   A_ij := A_ij + alpha x_i y_j
 *
 * where A is an m by n matrix, x is a vector of length m, y is
 * a vector of length n, and alpha is a constant. The data type
 * is double.
 *
 * Part 1. Replace the explicit cudaMalloc()/cudaMemcpy() by
 *         managed memory.
 * Part 2. Add prefetch requests for x and y before the kernel,
 *         and the matrix a after the kernel.
 * 
 * Copyright EPCC, The University of Edinburgh, 2023
 */

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

__host__ void myErrorHandler(cudaError_t ifail, const char * file,
                             int line, int fatal);

#define CUDA_ASSERT(call) { myErrorHandler((call), __FILE__, __LINE__, 1); }


/* Kernel parameters */

#define THREADS_PER_BLOCK_1D  256
#define THREADS_PER_BLOCK_2D   16

/* Kernel */

__global__ void myKernel(int mrow, int ncol, double alpha, double * x,
                         double * y, double * a) {

  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int i = blockIdx.y*blockDim.y + threadIdx.y;

  if (i < mrow && j < ncol) {
    a[i*ncol + j] = a[i*ncol + j] + alpha*x[i]*y[j];
  }

  return;
}

/* Main routine */

int main(int argc, char *argv[]) {

  int mrow = 1024;      /* Number of rows */
  int ncol =  512;      /* Number of columns */

  double alpha = 2.0;
  double * h_x = NULL;
  double * h_y = NULL;
  double * h_a = NULL;

  double * d_x = NULL;
  double * d_y = NULL;
  double * d_a = NULL;

  cudaStream_t stream;

  /* Check we have a GPU, and get device name from the cudaDeviceProp
   * structure. This is for information. */

  int ndevice = 0;
  int deviceNum = -1;
  cudaDeviceProp prop;

  CUDA_ASSERT( cudaGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
     printf("No GPU available!\n");
     exit(0);
  }

  CUDA_ASSERT( cudaGetDevice(&deviceNum) );
  CUDA_ASSERT( cudaGetDeviceProperties(&prop, deviceNum) );
  printf("Device %d name: %s\n", deviceNum, prop.name);
  printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);

  /* Initialise stream */

  CUDA_ASSERT( cudaStreamCreate(&stream) );

  /* Establish host data (with some initial values for x and y) */

  CUDA_ASSERT( cudaMallocHost(&h_x, mrow*sizeof(double)) );
  CUDA_ASSERT( cudaMallocHost(&h_y, ncol*sizeof(double)) );

  h_a = (double *) malloc(mrow*ncol*sizeof(double));
  assert(h_x);
  assert(h_y);
  assert(h_a);

  for (int i = 0; i < mrow; i++) {
    h_x[i] = 1.0*i;
  }
  for (int j = 0; j < ncol; j++) {
    h_y[j] = 1.0*j;
  }

  /* Establish device data and initialise A to zero on the device */
  /* Copy the initial values of x and y to device memory */

  CUDA_ASSERT( cudaMalloc(&d_x, mrow*sizeof(double)) );
  CUDA_ASSERT( cudaMalloc(&d_y, ncol*sizeof(double)) );

  cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  CUDA_ASSERT( cudaMemcpyAsync(d_x, h_x, mrow*sizeof(double), kind, stream) );
  CUDA_ASSERT( cudaMemcpyAsync(d_y, h_y, ncol*sizeof(double), kind, stream) );

  CUDA_ASSERT( cudaMalloc(&d_a, mrow*ncol*sizeof(double)) );
  CUDA_ASSERT( cudaMemset(d_a, 0, mrow*ncol*sizeof(double)) );

  CUDA_ASSERT( cudaStreamSynchronize(stream) );

  /* Define the execution configuration and run the kernel */

  unsigned int nblockx = 1 + (mrow - 1)/THREADS_PER_BLOCK_2D;
  unsigned int nblocky = 1 + (ncol - 1)/THREADS_PER_BLOCK_2D;
  dim3 blocks = {nblocky, nblockx, 1};
  dim3 threadsPerBlock = {THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D, 1};

  myKernel<<<blocks, threadsPerBlock>>>(mrow, ncol, alpha, d_x, d_y, d_a);

  CUDA_ASSERT( cudaPeekAtLastError() );
  CUDA_ASSERT( cudaDeviceSynchronize() );

  /* Retreive the results to h_a and check the results */

  kind = cudaMemcpyDeviceToHost;
  CUDA_ASSERT( cudaMemcpy(h_a, d_a, mrow*ncol*sizeof(double), kind) );

  int ncorrect = 0;
  printf("Results:\n");
  for (int i = 0; i < mrow; i++) {
    for (int j = 0; j < ncol; j++) {
      if (fabs(h_a[ncol*i + j] - alpha*h_x[i]*h_y[j]) < DBL_EPSILON) {
        ncorrect += 1;
      }
    }
  }
  printf("Number rows x cols %10d; correct: %10d\n", mrow*ncol, ncorrect);

  /* Release resources */

  CUDA_ASSERT( cudaFree(d_y) );
  CUDA_ASSERT( cudaFree(d_x) );
  CUDA_ASSERT( cudaFree(d_a) );
  CUDA_ASSERT( cudaStreamDestroy(stream) );

  CUDA_ASSERT( cudaFreeHost(h_y) );
  CUDA_ASSERT( cudaFreeHost(h_x) );

  free(h_a);

  return 0;
}

/* It is important to check the return code from API calls, so the
 * follow function/macro allow this to be done concisely as
 *
 *   CUDA_ASSERT(cudaRunTimeAPIFunction(...));
 *
 * Return codes may be asynchronous, and thus misleading! */

__host__ void myErrorHandler(cudaError_t ifail, const char * file,
                             int line, int fatal) {

  if (ifail != cudaSuccess) {
    fprintf(stderr, "Line %d (%s): %s: %s\n", line, file,
            cudaGetErrorName(ifail), cudaGetErrorString(ifail));
    if (fatal) exit(ifail);
  }

  return;
}
