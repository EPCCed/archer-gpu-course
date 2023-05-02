/*
 * This is an introductory exercise in using constant memory
 * and then shared memory. The reserve array idea is from an
*  NVIDIA exercise of the same type.
 *
 * Training material developed by Kevin Stratford
 * Copyright EPCC, The University of Edinburgh, 2017 
 */

#include <stdio.h>
#include <stdlib.h>

/* Forward Declaration*/
/* Utility function to check for and report CUDA errors */

void checkCUDAError(const char*);

/*
 * The number of CUDA threads per block to use.
 */

#define THREADS_PER_BLOCK 128

/* The number of integer elements in the array */

#define ARRAY_SIZE 65536

/* Reverse the elements in the input array d_in.
 * The total number of threads should be ARRAY_SIZE. */

__global__ void reverseArray(int * d_in, int * d_out)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  d_out[idx] = d_in[ARRAY_SIZE - (idx + 1)];
}


/* Main routine */
int main(int argc, char *argv[])
{
    int *h_in, *h_out;
    int *d_in, *d_out;

    int i;
    int ncorrect;
    size_t sz = ARRAY_SIZE * sizeof(int);

    /* Print device details */
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceNum);
    printf("  Device name: %s\n", prop.name);


    /*
     * allocate memory on host
     * h_in holds the input array, h_out holds the result
     */
    h_in = (int *) malloc(sz);
    h_out = (int *) malloc(sz);

    /*
     * allocate memory on device
     */
    cudaMalloc(&d_in, sz);
    cudaMalloc(&d_out, sz);

    /* initialise host arrays */
    for (i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = i;
        h_out[i] = 0;
    }

    /* copy input array from host to GPU */

    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);

    /* run the kernel on the GPU */

    dim3 blocksPerGrid(ARRAY_SIZE/THREADS_PER_BLOCK, 1, 1);
    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);

    reverseArray<<< blocksPerGrid, threadsPerBlock >>>(d_in, d_out);

    /* wait for all threads to complete and check for errors */

    cudaDeviceSynchronize();
    checkCUDAError("kernel invocation");

    /* copy the result array back to the host */

    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpyDeviceToHost");

    /* print out the result */
    printf("Results: ");
    ncorrect = 0;
    for (i = 0; i < ARRAY_SIZE; i++) {
      if (h_out[i] == h_in[ARRAY_SIZE - (i+1)]) ncorrect += 1;
    }
    printf("Number of correctly reversed elements %d (%s)\n", ncorrect,
           ncorrect == ARRAY_SIZE ? "Correct" : "INCORRECT");
    printf("\n");

    /* free device buffers */

    cudaFree(d_out);
    cudaFree(d_in);

    /* free host buffers */
    free(h_in);
    free(h_out);

    return 0;
}


/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
