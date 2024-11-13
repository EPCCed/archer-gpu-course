# Some performance considerations

We should now have a functioning, if simple, GPU program which moves
some data to and from the device and executes a kernel.

We can now ask the question: what factors can influence the
performance of such a program?


## Parallelism

Amdahl's law states that the parallel performance of a program is
limited by the fraction of code that is serial.

In the GPU context, this has two manifestations:

1. Kernel code is potentially parallel
2. Host code is definitely serial (including host-device transfers)

This may mean that additional work is required to expose parallelism,
or eliminate host operations in favour of device operations.

### Occupancy

Potentially, the GPU has a lot of SMs/cores that can be used. Having very
many blocks of work available at an one time is said to favour
high *occupancy*.

This may be thought of simply as having a very high degree of thread
parallelism. However, the degree is much higher than would be expected
on the basis of a threaded CPU program (where threads is usually the
number of cores).

Typically, a GPU kernel wants at least O(10^5) or O(10^6) threads to be
effective. That is, the problem space should have this number of elements.

If the problem does not have this natural degree of (data) parallelism,
there may be little benefit in using a GPU.

### Two dimensional example

Consider a two-dimensional loop:
```
   int NX = 512;
   int NY = 512;
   ...
   for (int i = 0; i < NX; i++) {
     for (int j = 0; j < NY; j++) {
       /* ... work for element (i, j) ... */
     }
   }
```

If we parallelised the inner loop only, we would have work for at most
512 threads. This would be two blocks if we were using, e.g., 256 threads
per blocks.

This would clearly be poor occupancy.


If we parallelised both loops, we would have 512 x 512 = 262,144 threads
(1024 blocks). This is much better. We now have a chance to employ many
SMs.

## Memory usage

### CPU: caching behaviour

A given thread in a CPU code favours consecutive memory accesses.
E.g., in C, recall that it is the right-most index that runs
fastest in memory.
```
   for (int i = 0; i < NX; i++) {
     for (int j = 0; j < NY; j++) {
       a[i][j] = 0.0;
     }
   }
```
Such an order displays favourable cache behaviour. A single thread makes
contiguous memory accesses.


### GPU: coalescing behaviour

For GPU global memory, the opposite is true. The hardware wants
to have warps of consecutive threads load consecutive memory
locations in a contiguous block.

Consider a one-dimensional example:
```
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  a_output[i] = a_input[i];
```
Here, there is no issue, consecutive threads (those with consecutive
x-index) access consecutive memory locations.


### Two dimensions again

Consider first:
```
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  for (int j = 0; j < NY; j++) {
    a_output[i][j] = a_input[i][j];
  }
```
Here, a given thread makes `NY` consecutive accesses to the arrays. This
does not favour coalesced access.

We want consecutive threads to have consecutive accesses, e.g.,
```
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  for (int i = 0; i < NX; i++) {
    a_output[i][j] = a_input[i][j];
  }
```

This is the favoured pattern. In other words, coalescing favours a given
thread making a strided memory access.


## Exercise (30 minutes)

The following exercise will examine the issue of parallelism and occupancy.
The current directory is a template `exercise_dger.cu` in which you
are asked to implement a kernel which computes the following matrix
operation:
```
  A_ij := A_ij + alpha x_i y_j
```
for a matrix A with m rows and n columns, a vector `x` of length m, a
vector `y` of length n, and constant `alpha`. The data type is
`double` in all cases.

For the matrix `a` we will adopt a flattened one-dimensional indexing
for which element row `i` and column `j` is addressed as `a[i*ncol + j]`.

As this is partly a performance issue (a correct answer is also required!)
we will implement some simple profiling by adding `nvprof` to the submission
script.

`nvprof` gives some basic text-based profile information for routines
involving the GPU at the end of execution. Try to keep a note of the time
taken by the kernel at each stage (reported in either milliseconds, `ms`,
or micro seconds, `us` by `nvprof`).


A suggested procedure is:
1. Check the template to see that the matrix and vectors have been established
   in device memory. Note that the template uses the CUDA API call
   ```
      cudaError_t cudaMemset(void * dptr, int value, size_t sz);
   ```
   to initialise all the device matrix elements to zero directly. The template
   should compile and run, but will not compute the correct answer as the
   kernel stub supplied does nothing.
2. Implement the most simple kernel in which the update is entirely
   serialised. E.g.,
   ```
   int tid = blockIdx.x*blockDim.x + threadIdx.x;

   if (tid == 0) {
     for (int i = 0; i < mrow; i++) {
       for (int j = 0; j < ncol; j++) {
          a[ncol*i + j] = a[ncol*i + j] + alpha*x[i]*y[j];
        }
     }
   }
   ```
   Check the execution configuration and run the code to ensure it reports
   the correct answer.

3. Eliminate the `i`-loop and re-check the kernel launch parameters to
   provide parallelism over rows of the matrix.
   Remember to allow that the problem size is not a whole number
   of blocks.

4. In addition, eliminate the `j`-loop to have parallelism over
   both rows and columns. You will need to introduce two dimensions
   in the abstract description, e.g., via
   ```
   int j = blockIdx.y*blockDim.y + threadIdx.y;
   ```
   and make an appropriate adjustment to the kernel launch parameters.
   Hint: keep the same total number of threads per block; but the block
   must become two-dimensional.

5. Is your resultant code getting the coalescing right? Consecutive
   threads, that is, threads with consecutive x-index, should
   access consecutive memory location.



### Finished?

If we had not used `cudaMemset()` to initialise the device values for
the matrix, what other options to initialise these values on the device
are available to us? (`cudaMemset()` is limited in that it can only be
used to initialise array values to zero, but not to other, non-zero, values.

For your best effort for the kernel, what is the overhead of the actual
kernel launch (`cudaLaunchKernel` in the profile) compared with the
time taken for the kernel itself?

What's the overhead for the host-device transfers?
