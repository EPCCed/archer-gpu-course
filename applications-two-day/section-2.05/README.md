# Shared memory

So far we have seen "global" memory, which is allocated by some
mechanism on the host, and is available in the kernel.

We have also used local variables in the kernel, which appear
on the stack in the expected fashion. Local variables are expected
to be held in registers and
take on a distinct value for each thread, e.g.,
```
  int i = blockDim.x*blockIdx.x + threadIdx.x;
```

While global memory is shared between all threads, the usage of
'shared memory' is reserved for something more specific.


## Independent accesses to global memory

In what we have seen so far, kernels have been used to replace
loops with independent iterations, e.g.,
```
  for (int i = 0; i < ndata; i++) {
    data[i] = 2.0*data[i];
  }
```
is replaced by a kernel with the body
```
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  data[i] = 2.0*data[i];
```
As each thread accesses an independent location in (global)
memory, there are no potential conflicts.

### A different pattern

Consider a loop with the following pattern
```
  double sum = 0.0;
  for (int i = 0; i < ndata; i++) {
    sum += data[i];
  }
```
The iterations are now coupled in some sense: all must accumulate
a value to the single memory location `sum`.

What would happen if we tried to run a kernel of the following
form?
```
  __global__ myKernel(int ndata, double * data, double * sum) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < ndata) *sum += data[i];
  }
```

### The problem

The problem lies in the fact that the increment is actually a
number of different operations which occur in order.
```
   *sum += ndata[i];
```
1. Read the current value of `sum` from memory into a register;
2. Undertake the appropriate arithmetic in register;
3. store the new result back to global memory.

If many threads are performing these operations in an uncontrolled
fashion, unexpected results can arise.

Such non-deterministic results are frequently referred to as race
conditions.

### The solution

In practice, potentially unsafe updates to any form of shared memory
must be protected by appropriate synchronisation: guarentees that
operations happen in the correct order.

For global memory, we require a so-called *atomic* update. For our
example above:
```
  *sum += data[i];            /* WRONG: unsafe update */
  atomicAdd(sum, data[i]);    /* Correct: atomic update */
```
Such updates are usually implemented by dome form of lock.

So the atomic update is a single unified operation on a single thread:
1. Obtain a lock on the relevant memory location (`sum`);
2. Read the existing value into register and update;
3. store the result back to the global memory location;
4. Release the lock on that location.


## Shared memory in blocks

There is an additional type of shared memory in kernels introduced
using the `__shared__` memory space quanlifer. E.g.,
```
  __shared__ double tmp[THREADS_PER_BLOCK];
```
These values are shared only between threads in the same block.


### Synchonisation

There are quite a large number of synchronisation options for
threads within a block in CUDA. The essential one is probably
```
  __syncthreads();
```
This is a barrier-like synchronisation which says that all
the threads in the block must reach the `__syncthreads()`
statement before any are allowed to continue.




## Exercise

In the following exercise we we implement a vector scalar product
in the style of the BLAS levle 1 routine `ddot()`.

The template provided sets up two vectors `x` and `y` with some
initial values. The exercise is to complete the `ddot()` kernel
which we will give prototype:
```
  __global__ void ddot(int n, double * x, double * result);
```
where the `result` is a single scalar value which is the dot
product.
