/*
 * Introduction.
 *
 * Implement the simple operation x := ax for a vector x of type double
 * and a constant 'a'.
 *
 * It introduces explicit device memory management.
 *
 * Part 1: declare and allocate device memory d_x
 * Part 2: copy host array h_x to device array d_x
 * Part 3: copy (unaltered) device array d_x back to the host array h_out
 * Part 4: remember to release device resources d_x at the end
 *
 * Training material originally developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010-2023
 */

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

/* Error checking routine and macro. */

__host__ void myErrorHandler(cudaError_t ifail, const char * file,
                             int line, int fatal);

#define CUDA_ASSERT(call) { myErrorHandler((call), __FILE__, __LINE__, 1); }


/* The number of integer elements in the array */
#define ARRAY_LENGTH 256

/* Suggested kernel parameters */
#define NUM_BLOCKS  1
#define THREADS_PER_BLOCK 256


/* Main routine */

int main(int argc, char *argv[]) {

  size_t sz = ARRAY_LENGTH*sizeof(double);

  double a = 1.0;          /* constant a */
  double * h_x = NULL;     /* input array (host) */
  double * h_out = NULL;   /* output array (host) */

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


  /* allocate memory on host; assign some initial values */

  h_x   = (double *) malloc(sz);
  h_out = (double *) malloc(sz);
  assert(h_x);
  assert(h_out);

  for (int i = 0; i < ARRAY_LENGTH; i++) {
    h_x[i] = 1.0*i;
    h_out[i] = 0;
  }

  /* allocate memory on device */

  /* copy input array from host to GPU */

  /* ... kernel will be here  ... */

  /* copy the result array back to the host output array */


  /* We can now check the results ... */
  printf("Results:\n");
  {
    int ncorrect = 0;
    for (int i = 0; i < ARRAY_LENGTH; i++) {
      /* The print statement can be uncommented for debugging... */
      /* printf("%9d %5.2f\n", i, h_out[i]); */
      if (fabs(h_out[i] - a*h_x[i]) < DBL_EPSILON) ncorrect += 1;
    }
    printf("No. elements %d, and correct: %d\n", ARRAY_LENGTH, ncorrect);
  }

  /* free device buffer */

  /* free host buffers */
  free(h_x);
  free(h_out);

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
