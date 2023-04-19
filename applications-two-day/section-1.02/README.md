
# The CUDA/HIP programming model

The very latest CUDA programming guide appears here. Note there may
be some features only supported on the very latest hardware.

https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## Picture



## Portability

CUDA has been under development by NVIDIA since around 2005. AMD, rather
later to the party, develops HIP, which is a shadow API for CUDA. For
example, a C/C++ call to 
```
  cudaMalloc(...);
```
is simply replaced by
```
  hipMalloc(...);
```
with the same signature. HIP code can be compiled for NVIDIA GPUs by
inclusion of an apprpropriate wrapper which just substitutes the relevant
CUDA API routine.

Not all the latest CUDA functionality is implemented in HIP at any given
time.

## Threads

## Workflow

Most applications require only the "runtime API" (C++). Lower level control
is offered by the "driver API" (C).

Kernels may be written in PTX, the CUDA assembly form, but it's much
easier to write C/C++. Both need to be compiled to a binary code for
execution. Note that the kernel language is a subset of C/C++.

Compilation separates host code and device code. Device code is
compiled to PTX and/or binary (cubin) forms. Modified host code
is typically then compiled using a host compiler.


## Compiler

The compiler is `nvcc` (not to be confused with `nvc` or `nvc++`).
This is usually provided as part of the NVIDIA HPC Toolkit.

CUDA code is often placed in files with the `.cu` extension
```
$ nvcc code.cu
```
One may prefer to use more standard file extensions, e.g., for
standard C:
```
$ nvcc -x cu code.c
```
where the `-x cu` option instructs `nvcc` to interpret code as CUDA C.


## Possible errors

E.g., the default `sm_52` will be applied if no `arch` is spcecified:
```
$ nvcc intro.cu
```
This will result in a run-time error of the form
```
cudaErrorUnsupportedPtxVersion: the provided PTX was compiled with an unsupported toolchain.
```
at the first device side operation - often `cudaMalloc()`.

On the other side, if the `sm` specification is too high, e.g.,
```
$ nvcc -arch=sm_90
```
then one will obtain a
```
cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device
```
at the point of kernl launch.
