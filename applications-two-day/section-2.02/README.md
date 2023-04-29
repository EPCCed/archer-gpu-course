# Kernels

In many scientific applications, a large fraction of the computational
effort is spent in *kernels*. It is natural that kernels should then be
the first target for moving to the GPU.

Kernels are often associated with loops for which we can distribute
independent iterations between threads in parallel.


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
```
The `__global__` is a so-called execution space qualifier and
indicates to the compiler that this is an entry point for
GPU execution. The `threadIdx.x` is a special variable that CUDA
provides to allow us to identify the thread.

The second step is to execute, or *launch*, the kernel on the GPU.
This is done by specifying the number of blocks, and the number
of threads per block:
```
  dim3 blocks = {1, 1, 1};
  dim3 threadsPerBlock = {ARRAY_LENGTH, 1, 1};

  myKernel<<<blocks, threadsPerBlock>>>(result);
```
The language extension `<<<...>>>` is referred to as the
execution configuration. It comes between the kernel
function name, and the arguments.

We must arrange that the product of the number of blocks and the
number of threads per block give the correct total number of
threads for the problem at hand (i.e., `ARRAY_LENGTH`). This is
only valid is `ARRAY_LENGTH` is the number of threads per block.

We have introduced the structure
```
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
  } dim3;
```
which may be intialised in C as above, or using C++ style
constructors.


### More than one block

If we want to have a large problem, we need more blocks. Usually, for
a large array, we want very many blocks, e.g.,
```
__global__ void myKernel(int * result) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  result[i] = 2*i;
}
```
would could have execution configuration
```
  threadsPerBlock.x = THREADS_PER_BLOCK;
  blocks.x          = ARRAY_LENGTH/THREADS_PER_BLOCK;

  myKernel<<< blocks, threadsPerBlock >>>(result);
```

This is the more general modus operandi:
1. Choose a number of threads per block (128, 256, ..., 1024)
2. Compute the number of blocks required to cover the problem space.

In the above, we still assume that `ARRAY_LENGTH` is exactly
divisible by `THREADS_PER_BLOCK`.


### Internal variables in the kernel

A number of internal variables are made available by the CUDA
runtime and can be used in the kernel to locate a given
thread's position in the abstract grid picture:
```
   dim3 gridDim;     /* The number of blocks */
   dim3 blockDim;    /* The number of threads per block */
   
   /* Unique to each block: */
   dim3 blockIdx;    /* 0 <= blockIdx.x < gridDim.x  etc. for y,z */

   /* Unique to each thread (within a block): */
   dim3 threadIdx;   /* 0 <= threadIdx.x < blockDim.x  etc. for y,z */
```
These names should be considered reserved.


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

Errors occuring in the kernel execution are also asynchronous, which
can cause some confusion. As a result, one will sometimes see this
usage:
```
   myKernel<<< blocks, threadsPerBlock >>>{arg1, arg2, arg3);

   CUDA_ASSERT( cudaPeekAtLastError() );
   CUDA_ASSERT( cudaDeviceSynchronize() );
```

The first function, `cudaPeekAtLastError()`, queries the error state
without altering it, and will fail if there are errors in the kernel
launch (e.g., the GPU is not available, or there are errors
in the configuration).

Errors that occur during the execution of the kernel itself will not
be apparent until `cudaDeviceSynchronize()`. This may typically be
a programmer error which will need to be debugged.


## Exercise (cont.)

Starting from your solution to the previous exercise, we will now
add the relevant kernel and execution configuration. You should
adjust the value of the constant `a` to be e.g., `a = 2.0`.

There is also a new template with a canned solution to the previous
part in this directory.

### Sugggested procedure

1. Write a kernel of the prototype
```
__global__ void myKernel(double a, double * x);
```
to perform the scale operation on a given element of the array.
Limit yourself to one block in the first instance (you only
need `threadIdx.x`).

2. In the main part of the program, declare and initialise
variables of type `dim3` to hold
the number of blocks, and the number of threads per block.
You can use one block and `THREADS_PER_BLOCK` in the first
instance.

### More than one block

Update the kernel, and then the execution configuration parameters
to allow more than one block. We will keep the assumption that
the array length is a whole number of blocks.

Increase the array size `ARRAY_LENGTH` to 512, and check you retain
the correct behaviour. Check for larger multiples of
`THREADS_PER_BLOCK`.

### Problem size not a whole number of blocks

As we are effectively contrained in the choice of `THREADS_PER_BLOCK`,
it is likely that the problem space is not an integral number of
blocks for general problems. How can we deal with this situation?

1. For the launch parameters, you will need to compute a number of blocks
that is sufficient and necessary to cover the entire problem space. (There
needs to be at least one block, but no more than necesary.)
2. You will also need to make an adjustment in the kernel. To avoid what
type of error?

Set the array size to, e.g., 100, and then to 1000 to check your result.


### Finished?

All kernels must be declared `void`. Why do you think this is the case?


Adapt your program to try another simple level one BLAS routine, which
we will take to have the prototype:
```
  void daxpy(int nlen, double a, const double * x, double * y);
```
This is the operation `y := ax + y` where `y` is incremented, and `x` is
unchanged. Both vectors have length `nlen`.


**Expert point**. If you are not keen on the non-standard looking execution
configuration ```<<<...>>>```, one can also use the C++ API function
`cudaLaunchKernel()`. However, this is slightly more awkward.

The prototype expected is:
```
  cudaErr_t cudaLaunchKernel(const void * func, dim3 blocks, dim3 threads, void ** args, ...);
```
where only the first 4 arguments are required. The kernel function is `func`, while
the second and third arguments are the number of blocks and threads per block
(taking the place of the execution configuration). The fourth argument holds the
kernel parameters. Hint: for our first kernel this final argument will be
```
   void *args[] = {&a, &d_x};
```
As `cudaLaunchKernel()` is an API function returning an error, the return code can be
inpsected with the macro to check for errors in the launch (instead of `cudaPeekAtLastError()`). 
