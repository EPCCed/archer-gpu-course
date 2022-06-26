/*
 * Exercise.
 * Compute the vector product of two double arrays a and b.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010 
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char*);

/* The number of integer elements in the array */
#define ARRAY_SIZE 2048

/*
 * The number of CUDA blocks and threads per block to use.
 */

#define THREADS_PER_BLOCK 256

/* The kernel is here: */

__global__ void vector_product(double * a, double * b, double * result) {

/* PART 2: implemnet the kernel to compute the correct result */

}

/* Main routine */

int main(int argc, char *argv[]) {

  size_t sz = ARRAY_SIZE*sizeof(double);

  double * h_a = NULL;
  double * h_b = NULL;
  double * d_a = NULL;
  double * d_b = NULL;

  /* Print device details */

  int nDevice = 0;

  cudaGetDeviceCount(&nDevice);
  if (nDevice == 0) {
    printf("No devices detected. Make sure you run in the queue system.\n");
    exit(-1);
  }
  else {
    int deviceNum;
    cudaDeviceProp prop;
    cudaGetDevice(&deviceNum);	
    cudaGetDeviceProperties(&prop, deviceNum);
    printf("  Device name: %s\n", prop.name);
  }   

  /*
   * allocate memory on host
   */

  h_a = (double *) malloc(sz);
  h_b = (double *) malloc(sz);

  /*
   * allocate memory on device
   */

  cudaMalloc((void **) &d_a, sz);
  cudaMalloc((void **) &d_b, sz);

  /* initialise host arrays */

  for (int i = 0; i < ARRAY_SIZE; i++) {
    h_a[i] = 1.0;
    h_b[i] = 2.0;
  }

  cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);

  /* PART 1: allocate and initialise device result */
  
  /* run the kernel on the GPU */

  {
    dim3 nthreads = THREADS_PER_BLOCK;	
    dim3 nblocks  = 1 + (ARRAY_SIZE - 1)/THREADS_PER_BLOCK;
    /* PART 2: Kernel */
  }   

  /* wait for all threads to complete and check for errors */

  cudaDeviceSynchronize();
  checkCUDAError("kernel invocation");

  /* Result */

  printf("Results:\n");

  {
    double h_sum = 0.0;
    double h_sum_expect = 0.0;

    /* PART 1: Copy result back to host here ... */

    for (int i = 0; i < ARRAY_SIZE; i++) {
      h_sum_expect += h_a[i]*h_b[i];
    }
    printf("Host result:   %14.7e\n", h_sum_expect);
    printf("Kernel result: %14.7e\n", h_sum);
  }
  printf("\n");

  /* free device buffers */

  cudaFree(d_b);
  cudaFree(d_a);

  /* free host buffers */

  free(h_a);
  free(h_b);

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
