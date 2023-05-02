/*
 * Use of shared memory and atomic updates.
 *
 * An implementation of the blas-like level 1 routine ddot(), which is
 * the vector scalar product
 *
 *   res = x_n y_n
 *
 * where x and y are both vectors (of type double) and of length n elements.
 *
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

#define THREADS_PER_BLOCK 256

__global__ void ddot(int n, double * x, double * y, double * result) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid == 0) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
      sum += x[i]*y[i];
    }
    *result = sum;
  }

  return;
}

/* Main routine */

int main(int argc, char *argv[]) {

  int nvector = 2048*64;    /* Length of vectors x, y */

  double * h_x = NULL;
  double * h_y = NULL;
  double * d_x = NULL;
  double * d_y = NULL;

  double h_result = 0.0;
  double * d_result = NULL;

  /* Establish host data (with some initial values for x and y) */

  h_x = (double *) malloc(nvector*sizeof(double));
  h_y = (double *) malloc(nvector*sizeof(double));
  assert(h_x);
  assert(h_y);

  for (int i = 0; i < nvector; i++) {
    h_x[i] = 1.0*i;
  }
  for (int j = 0; j < nvector; j++) {
    h_y[j] = 2.0*j;
  }

  /* Establish device data and initialise A to zero on the device */
  /* Copy the initial values of x and y to device memory */
  /* Also need device memory for the (scalar) result */

  CUDA_ASSERT( cudaMalloc(&d_x, nvector*sizeof(double)) );
  CUDA_ASSERT( cudaMalloc(&d_y, nvector*sizeof(double)) );

  cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  CUDA_ASSERT( cudaMemcpy(d_x, h_x, nvector*sizeof(double), kind) );
  CUDA_ASSERT( cudaMemcpy(d_y, h_y, nvector*sizeof(double), kind) );

  CUDA_ASSERT( cudaMalloc(&d_result, sizeof(double)) );

  /* Define the execution configuration and run the kernel */

  unsigned int nblockx = 1 + (nvector - 1)/THREADS_PER_BLOCK;
  dim3 blocks = {nblockx, 1, 1};
  dim3 threadsPerBlock = {THREADS_PER_BLOCK, 1, 1};

  ddot<<<blocks, threadsPerBlock>>>(nvector, d_x, d_y, d_result);

  CUDA_ASSERT( cudaPeekAtLastError() );
  CUDA_ASSERT( cudaDeviceSynchronize() );

  /* Retrieve the result and check. */

  kind = cudaMemcpyDeviceToHost;
  CUDA_ASSERT( cudaMemcpy(&h_result, d_result, sizeof(double), kind) );

  double result = 0.0;
  for (int i = 0; i < nvector; i++) {
    result += h_x[i]*h_y[i];
  }
  printf("Result for device dot product is: %14.7e (correct %14.7e)\n",
         h_result, result);
  if (fabs(h_result - result) < DBL_EPSILON) {
    printf("Correct.\n");
  }
  else {
    printf("FAIL!\n");
  }

  /* Release resources */

  CUDA_ASSERT( cudaFree(d_y) );
  free(h_y);
  free(h_x);

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
