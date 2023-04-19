# CUDA Programming

The first topic we must address is the existance of separate address
spaces for CPU and GPU memory, and moving data between them.

It may be useful to look at the CUDA runtime API reference:
https://docs.nvidia.com/cuda/cuda-runtime-api/index.html


## What to include and what not to include

A standard C/C++ source file may include
```
#include "cuda_runtime.h"
```
which is usually relevant for programs to be compiled by `nvcc`.


There is also a subset
```
#include "cuda_runtime_api.h"
```
which is the C interface which does not need to be compiled with `nvcc`.

There is also
```
#include "cuda.h"
```
which is the CUDA driver API (a lower level interface). We will not
consider the driver API in this course.


## Memory management

Data accessed bh kernels must reside in device memory, sometimes also
referred to as device global memory, or just "global memory".

There are different ways of managing the allocation and movement
of data between host and device. Broadly:

1. explicit allocations and exxplicit copies
2. use of 'managed' memory

We will look at the explicit mechanism first.


## Memory Allocation

Declaration is via standard C data types and pointers, e.g.,

```
  double * data = NULL;   /* Device data */

  err = cudaMalloc(&data, nArray*sizeof(double));

  ...

  err = cudaFree(data);
```

Such pointers are "host pointers to device memory". They have a value,
but cannot be dereferred in host code.

We will return to error handling below.

## Memory movement

Assuming we have established some data on the host, copies are
via `cudaMemcpy()`. Schematically,
```
  err = cudaMemcpy(data, hostdata, nArray*sizeof(double), cudaMemcpyHostToDevice);

  /* ... do seomthing ... */

  err = cudaMemcpy(hostdata, data, nArray*sizeof(double), cudaMemcpyDeviceToHost);
```

Formally, the API reads

cudaError_t cudaMemcpy(void * dest, void * src, size_t sz, cudaMemcpyKind direction);


## Error handling

Most CUDA API routines return an error code of type `cudaError_t`.
It is important to check the return value against `cudaSuccess`.

If an error occurs, the error code can be interrogated to provide
some meaningful information. E.g. use
```
const char * cudaGetErrorName(cudaError_t err);    /* Name */
const char * cudaGetErrorString(cudaError_t err);  /* Descriptive string */
```

## Error handling in practice

The requirement for error handling is often handled in real C code
using a macro, e.g.,
```
  CUDA_ASSERT( cudaMalloc(&data, nArray*sizeof(double) );
```

To avoid clutter, we omit this error checking in the example
code snippets.

However, for the code exercises, we have provided such a macro, and
it should be used.

It is particularly important to check the result of the first API
call in the code. This will detect any problems with the CUDA
context, and may avoid surprises later in the code.


## Exercise (20 minutes)

Look at the associated exercise `intro.cu`. This provides a template
for a first exercise which is to implement a simple scale function,
which will multiply all the elements of an array by a constant.

The first part of the exercise is to allocate and move data to and
from the GPU. We will address the kernel in the next exercise.

First, check you can compile and run the unaltered template code in
the queue system.

Second, undertake the following steps:

1A. declare and allocate device memory (call it `d_x`) of type `double`;
1B. copy the initialised host array `h_x` to device array `d_x`
1C. copy the (unaltered) device array `d_x` back to the host array `h_out`
    and check that `h_out` has the expected values;
1D. release the device resources `d_x` at the end of execution.

