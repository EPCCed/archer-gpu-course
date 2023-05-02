/*
 * Matrix vector product
 *
 * Here we will implement a slightly simplified version of the BLAS
 * level 2 routine dgemv() which computes the matrix-vector product
 *
 *    y_i := beta*y_i + alpha*A_ij x_j
 *
 * for an m x n matrix A_mn and vectors x_n and y_m. The data type
 * is double, with alpha and beta sclar constants.
 *
 * The simplification is that we will consider only
 *
 *    y_i := alpha*A_ij x_j
 *
 * Again we will assume that we are going to address the matrix with
 * the flattened one-dimensional index A_ij = a[i*ncol + j] with ncol
 * the number of columns n.
 *
 * An extirely serial kernel is provided below.
 *
 * Training material developed by Nick Johnson and Kevin Stratford
 * Copyright EPCC, The University of Edinburgh, 2010-2023
 */

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

__host__ void myErrorHandler(cudaError_t ifail, const char * file,
                             int line, int fatal);

#define CUDA_ASSERT(call) { myErrorHandler((call), __FILE__, __LINE__, 1); }

/* Kernel parameters (start with 1-d) */

#define THREADS_PER_BLOCK   256
#define THREADS_PER_BLOCKX   16
#define THREADS_PER_BLOCKY   16

/* An entirely serial kernel. */
__global__ void myKernel0(int mrow, int ncol, double alpha, double * a,
                          double * x, double * y) {

  int tid = blockIdx.x*blockDim.x  + threadIdx.x;

  if (tid == 0) {
    for (int i = 0; i < mrow; i++) {
      double sum = 0.0;
      for (int j = 0; j < ncol; j++) {
        sum += a[i*ncol + j]*x[j];
      }
      y[i] += alpha*sum;
    }
  }
  return;
}

/* After step 1 in the instructions */

__global__ void myKernelStep1(int mrow, int ncol, double alpha, double * a,
                              double * x, double * y) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;

  int tidj = threadIdx.y;

  __shared__ double tmp[THREADS_PER_BLOCK];

  assert(gridDim.x  == mrow);
  assert(blockDim.x ==    1);

  tmp[tidj] = 0.0;
  if (i < mrow && j < ncol) {
    tmp[tidj] += a[i*ncol + j]*x[j];
  }

  __syncthreads();

  /* Thread 0 in each row (block) can accumulate the contribution */
  if (tidj == 0) {
    double sum  = 0.0;
    for (int it = 0; it < blockDim.y; it++) {
      sum += tmp[it];
    }
    atomicAdd(y + i, alpha*sum);
  }

  return;
}

/* After step 2 in the instructions */


__global__ void myKernelStep2(int mrow, int ncol, double alpha, double * a,
                              double * x, double * y) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  int tidx = threadIdx.x;
  int tidy = threadIdx.y;

  __shared__ double tmp[THREADS_PER_BLOCKX][THREADS_PER_BLOCKY];

  tmp[tidx][tidy] = 0.0;

  if (i < mrow && j < ncol) {
    tmp[tidx][tidy] = a[i*ncol + j]*x[j];
  }

  __syncthreads();

  /* One thread per block accumulates per row ... */
  if (tidy == 0) {
    double sumrow = 0.0;
    for (int ity = 0; ity < blockDim.y; ity++) {
      sumrow += tmp[tidx][ity];
    }

    atomicAdd(y + i, alpha*sumrow);
  }

  return;
}

/* Step 3: And rotate the indices to admit coalescing */

__global__ void myKernelStep3(int mrow, int ncol, double alpha, double * a,
                              double * x, double * y) {

  /* Retain (i,j) as (row,column) of matrix but let j run fastest */
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int i = blockIdx.y*blockDim.y + threadIdx.y;

  int tidj = threadIdx.x;
  int tidi = threadIdx.y;

  __shared__ double tmp[THREADS_PER_BLOCKY][THREADS_PER_BLOCKX];

  tmp[tidi][tidj] = 0.0;

  if (i < mrow && j < ncol) {
    tmp[tidi][tidj] = a[i*ncol + j]*x[j];
  }

  __syncthreads();

  /* One thread per block accumulates per row ... */
  if (tidj == 0) {
    double sumrow = 0.0;
    for (int it = 0; it < blockDim.x; it++) {
      sumrow += tmp[tidi][it];
    }

    atomicAdd(y + i, alpha*sumrow);
  }

  return;
}



/* Main routine */

int main(int argc, char *argv[]) {

  int mrow = 1024;       /* Number of rows */
  int ncol =  256;       /* Number of columns (start = THREADS_PER_BLOCK) */

  double alpha = 2.0;

  double * h_x = NULL;
  double * h_y = NULL;
  double * d_x = NULL;
  double * d_y = NULL;
  double * h_a = NULL;
  double * d_a = NULL;

  /* Print device name (just for information) */

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
  printf("Device name: %s\n", prop.name);
  printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);

  /*
   * Establish some data on the host. x = 1; y = 0; A_ij = 1
   */

  h_x = (double *) malloc(ncol*sizeof(double));
  h_y = (double *) malloc(mrow*sizeof(double));
  h_a = (double *) malloc(mrow*ncol*sizeof(double));
  assert(h_x);
  assert(h_y);
  assert(h_a);

  for (int i = 0; i < ncol; i++) {
    h_x[i] = 1.0;
  }
  for (int j = 0; j < mrow; j++) {
    h_y[j] = 0.0;
  }
  for (int i = 0; i < mrow; i++) {
    for (int j = 0; j < ncol; j++) {
      h_a[i*ncol + j] = 1.0*i*j;
    }
  }

  /*
   * allocate memory on device
   */

  CUDA_ASSERT( cudaMalloc(&d_x, ncol*sizeof(double)) );
  CUDA_ASSERT( cudaMalloc(&d_y, mrow*sizeof(double)) );
  CUDA_ASSERT( cudaMalloc(&d_a, mrow*ncol*sizeof(double)) );

  cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  CUDA_ASSERT( cudaMemcpy(d_x, h_x, ncol*sizeof(double),      kind) );
  CUDA_ASSERT( cudaMemcpy(d_y, h_y, mrow*sizeof(double),      kind) );
  CUDA_ASSERT( cudaMemcpy(d_a, h_a, mrow*ncol*sizeof(double), kind) );


  /* Kernel */

  unsigned int nblocky = 1 + (mrow - 1) / THREADS_PER_BLOCKX;
  unsigned int nblockx = 1 + (ncol - 1) / THREADS_PER_BLOCKY;
  dim3 blocks = {nblockx, nblocky, 1};
  dim3 threadsPerBlock = {THREADS_PER_BLOCKX, THREADS_PER_BLOCKY, 1};

  myKernelStep3<<<blocks, threadsPerBlock>>>(mrow, ncol, alpha, d_a, d_x, d_y);

  /* wait for all threads to complete and check for errors */

  CUDA_ASSERT( cudaPeekAtLastError() );
  CUDA_ASSERT( cudaDeviceSynchronize() );

  kind = cudaMemcpyDeviceToHost;
  CUDA_ASSERT( cudaMemcpy(h_y, d_y, mrow*sizeof(double), kind) );

  printf("Results:\n");
  {
    int ncorrect = 0;
    for (int i = 0; i < mrow; i++) {
      double sum = 0.0;
      double yi  = 0.0;
      for (int j = 0; j < ncol; j++) {
        sum += h_a[i*ncol + j]*h_x[j];
      }
      yi = alpha*sum;
      if (fabs(yi - h_y[i]) < DBL_EPSILON) ncorrect += 1;
      /* Can be uncommented to debug ... */
      /* printf("Row %5d %14.7e %14.7e\n", i, y, h_y[i]); */
    }
    printf("No. rows %d, and correct rows %d\n", mrow, ncorrect);
  }

  /* free device buffer */
  CUDA_ASSERT( cudaFree(d_a) );
  CUDA_ASSERT( cudaFree(d_y) );
  CUDA_ASSERT( cudaFree(d_x) );

  /* free host buffers */
  free(h_x);
  free(h_y);
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
