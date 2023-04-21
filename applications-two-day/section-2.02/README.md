# Kernels

## A simple example

Consider the following loop in C:
```
for (int i = 0; i < ARRAY_LENGTH; i++) {
  result[i] = 2*i;
}
```
As each iteration of the loop is independent, we can safely
attempt to run each iteration in parallel.

In CUDA, we would we need to take two steps. First, we write
a kernel function which expresses the body of the loop:
```

__global__ void myKernel(int * result) {

  int i = threadIdx.x;

  result[i] = 2*i;
}

The second step is to execute, or *launch*, the kernel on the GPU.
This is done by specifying the number of blocks, and the number
of threads per block:
```
  dim3 blocks = {1, 1, 1};
  dim3 threadsPerBlock = {LOOP_LENGTH, 1, 1};

  myKernel<<<blocks, threadsPerBlock>>>(result);
```
Here we request 1 block of `LOOP_LENGTH` threads. If we assume that
`LOOP_LENGTH` is no larger than the maximum number of threads per
block, then this is ok. (As this is a one-dimensional problem,
we only care about the x-dimension; others are set to unity.)

We have introduced the structure
```
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
  } dim3;
```
which may be intialised in C as above, or using C++ constructors.


## More than one block

If we wish to use more than the maximum number of threads per block,
we need more blocks. Usually, for a large array, we want very many
blocks, e.g.,
```
__global__ void myKernel(int * result) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  result[i] = 2*i;
}
```
would could have execution configuration
```
  threadsPerBlock.x = THREADS_PER_BLOCK;
  block.x           = LOOP_LENGTH/THREADS_PER_BLOCK;

  myKernel<<< blocks, threadsPerBlock >>>(result);
```

## Internal variables


## Synchronisation between host and device

Kernel launches are asynchronous on the host. The launch will return
immediately. In order to be sure that the kernel has actually
completed, we need synchronisation.
```

  myKernel<<< blocks, threadsPerBlock >>>(arg1, arg2, arg3);

  /* ... returns immediately */

  cudaErr_t err = cudaDeviceSynchronize();

  /* ... it is now safe to use the results of the kernel ... */
```


### Error handling

Errors occuring in the kernel execution are also asynchronoues, which
can cause some confusion. As a result, one will sometimes see this
usage:
```
   myKernel<<< blocks, threadsPerBlock >>>{arg1, arg2, arg3);

   CUDA_ASSERT( cudaPeekAtLastError() );
   CUDA_ASSERT( cudaDeviceSynchronize() );
```


## Exercise (cont.)

Starting from your solution to the previous exercise, we will now
add the relevant kernel and execution configuration.

(There is also a new template with a canned solution to the previous
part in this directory.)

