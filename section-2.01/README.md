# CUDA Programming

The first topic we must address is the existence of separate address
spaces for CPU and GPU memory, and moving data between them.

![Schematic of host/device memories](../images/ks-schematic-memory-transfer.svg)


We need to take appropriate action in our code.


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

C programmers: C must be the subset of C which is also valid C++ to
use `nvcc`.


There is also
```
#include "cuda.h"
```
which is the CUDA driver API (a lower level interface). We will not
consider the driver API in this course. (CUDA driver API routines
are of the form `cuDeviceGet()`.)

## Context

There is no explicit initialisation required in the code. The first
call to the CUDA API will cause the CUDA context to be initialised
behind the scenes.



## Memory management

Data accessed by kernels must reside in device memory, sometimes also
referred to as device global memory, or just "global memory".

There are different ways of managing the allocation and movement
of data between host and device. Broadly:

1. explicit allocations and explicit copies;
2. use of 'managed' memory.

We will look at the explicit mechanism first.


## Memory Allocation

Declaration is via standard C data types and pointers, e.g.,

```
  double * data = NULL;   /* Device data */

  err = cudaMalloc(&data, nArray*sizeof(double));

  /* ... perform some work ... */

  err = cudaFree(data);
```

Such pointers are "host pointers to device memory". They have a value,
but cannot be dereferrenced in host code (a programmer error).

We will return to error handling below.

## Memory movement

Assuming we have established some data on the host, copies are
via `cudaMemcpy()`. Schematically,
```
  err = cudaMemcpy(data, hostdata, nArray*sizeof(double),
                   cudaMemcpyHostToDevice);

  /* ... do something ... */

  err = cudaMemcpy(hostdata, data, nArray*sizeof(double),
                   cudaMemcpyDeviceToHost);
```

These are *blocking* calls: they will not return until the data has been
stored in GPU memory (or an error has occurred).

Formally, the API reads
```
cudaError_t cudaMemcpy(void * dest, void * src, size_t sz,
                       cudaMemcpyKind direction);
```

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

The requirement for error handling often appears in real C code
as a macro, e.g.,
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

Look at the associated exercise `exercise_dscal.cu`. This provides a template
for a first exercise which is to implement a simple scale function,
which will multiply all the elements of an array by a constant.

The first part of the exercise is to allocate and move data to and
from the GPU. We will address the kernel in the next exercise.

First, check you can compile and run the unaltered template code in
the queue system.

Recall that we should use
```
$ nvcc -arch=sm_70 exercise_dscal.cu
```
and submit to the queue system using the script provided. If the code has run
correctly, you should see in the output something like:
```
Device 0 name: Tesla V100-SXM2-16GB
Maximum number of threads per block: 1024
Results:
No. elements 256, and correct: 1
```
showing that only one output array element is correct.

Second, undertake the following steps:

1. declare and allocate device memory (call it `d_x`) of type `double`;
2. copy the initialised host array `h_x` to device array `d_x`
3. copy the (unaltered) device array `d_x` back to the host array `h_out`
    and check that `h_out` has the expected values;
4. release the device resources `d_x` at the end of execution.

Remember to use the macro to check the return value of each of the CUDA
API calls.

As there is no kernel yet, the output `h_out` should just be the same
as the input `h_in` if operating correctly. All 256 array elements
should be correct.

### Finished?

Check the CUDA documentation to see what other information is available
from the structure `cudaDeviceProp`. This will be in the section on
device management in the CUDA runtime API reference.

What other possibilities exist for `cudaMemcpyKind`?

https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

### What can go wrong?

What happens if you forget the `-arch=sm_70` in the compilation?

What happens if you mess up the order of the host and device references in
a call to `cudaMemcpy()`? E.g.,
```
  cudaMemcpy(hostdata, devicedata, sz, cudaMemcpyHostToDevice);
```
where the data has been allocated as the names suggest.
