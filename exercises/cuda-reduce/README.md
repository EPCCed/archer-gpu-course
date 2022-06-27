# Introduction: A simple reduction

## Credits

Exercise created by EPCC, The University of Edinburgh. Documentation and
source code copyright The University of Edinburgh 2022.

Material by: Kevin Stratford

## Purpose

This exercise demostrates how to perform a simple reduction within a
kernel. We consider a scalar vector product, a standard C version
of which might be, schematically:
```
double sum = 0.0;
for (int i = 0; i < ARRAY_SIZE; i++) {
  sum += a[i]*b[i];
}
```
where `a[]` and `b[]` are arrays of length `ARRAY_SIZE`.

## Source code

You can get this from GitHub:

```
git clone https://github.com/EPCCed/archer-gpu-course.git
cd archer-gpu-course/exercises/cuda-intro/
```

The template source file is clearly marked with the sections to be
edited, e.g.

      /* Part 1A: allocate device memory */
      

Where necessary, you can refer to the CUDA C Programming Guide and
Reference Manual documents available from
<http://developer.nvidia.com/nvidia-gpu-computing-documentation>.


## Part 1: Allocate device memory for the result

As, ultimately, we want to access the final result on the host, we
must allocate device memory for the (single) `double` value on the
device. The relevant pointer will be passed to to the kernel.

The device result variable should be initialised to zero. This can
be done once on the host using `cudaMemset()`.

Finally the result must be copied back to the host to check the
result.

## Part 2: Implement the kernel

Declare an array of shared variables, one for each thread in the block.
Each block then assigns the relevant contribution to the sum to the
appropriate element of this array.

When all elements of the shared array have been assigned, a single thread
can then form the sum of all the contributions within one block. (This is
usually done on thread 0.)

Check this works by invoking a kernel with a single block.

For the general case with more than one block, an atomic operation is
required to update the final result safely with the contriubtion from
each block. Add this and check the general case returns the correct
answer.


## Compilation

Load the NVIDIA HPCSDK module to allow compilation:

```shell
module load nvidia/nvhpc
```

Compile the code using `make`. Note that the compute capability of the
CUDA device is specified with the `-arch` flag for C.


