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
'shared memory' is reserved for something more specific in the
GPU context. This is discussed below.


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
must be protected by appropriate synchronisation: guarantees that
operations happen in the correct order.

For global memory, we require a so-called *atomic* update. For our
example above:
```
  *sum += data[i];            /* WRONG: unsafe update */
  atomicAdd(sum, data[i]);    /* Correct: atomic update */
```
Such updates are usually implemented by some form of lock.

So the atomic update is a single unified operation on a single thread:
1. obtain a lock on the relevant memory location (`sum`);
2. read the existing value into register and update;
3. store the result back to the global memory location;
4. release the lock on that location.

### Note

`atomicAdd()` is an overloaded device function:
```
__device__ int atomicAdd(int * address, int value);
__device__ double atomicAdd(double * address, double value);
```
and so on. The old value of the target variable is returned.


## Shared memory in blocks

There is an additional type of shared memory available in kernels
introduced using the `__shared__` memory space qualifier. E.g.,
```
  __shared__ double tmp[THREADS_PER_BLOCK];
```
These values are shared only between threads in the same block.

Potential uses:
1. marshalling data within a block;
2. temporary values (particularly if there is significant reuse);
3. contributions to reduction operations.

Note: in the above example we have fixed the size of the `tmp`
object at compile time ("static" shared memory).

### Synchronisation

There are quite a large number of synchronisation options for
threads within a block in CUDA. The essential one is probably
```
  __syncthreads();
```
This is a barrier-like synchronisation which says that all
the threads in the block must reach the `__syncthreads()`
statement before any are allowed to continue.


### Example
Here is a (slightly contrived) example:
```
/* Reverse elements so that the order 0,1,2,3,...
 * becomes ...,3,2,1,0
 * Assume we have one block. */

__global__ void reverseElements(int * myArray) {

  __shared__ int tmp[THREADS_PER_BLOCK];

  int idx = threadIdx.x;
  tmp[idx] = myArray[idx];

  __syncthreads();

  myArray[THREADS_PER_BLOCK - (idx+1)] = tmp[idx];
}
```

### Synchronisation hazards

The usual conisderations apply when thinking about thread
synchronisation. E.g.,
```
   if (condition) {
      __syncthreads();
   }
```
There is a potential for deadlock.

### Branch divergence

It is beneficial for performance to avoid "warp divergence"
e.g.,
```
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (tid % 2 == 0) {
    /* threads 0, 2, 4, ... */
  }
  else {
    /* threads 1, 3, 5 ... *.
  }
```
may cause seralisation. For this reason you may see things
like
```
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  
  if ((tid / warpSize) % 2 == 0) {
     /* threads 0, 1, 2, ... */
  }
  else {
     /* threads 32, 33, 34, ... */
  }
```
where `warpSize` is another special value provided by CUDA.

## Other potential performance concerns

Shared memory via `__shared__` is a finite resource. The exact amount
will depend on the particular hardware, but may be in the region of
64 KB. (A portable program might have to take action at run time to
control this: e.g., using "dynamic" shared memory, where the size is
set as a kernel launch parameter.)

If an individual block requires a large amount of shared memory, then
this may limit the number of blocks that can be scheduled at the same
time, and so harm occupancy.


## Exercise (20 minutes)

In the following exercise we we implement a vector scalar product
in the style of the BLAS level 1 routine `ddot()`.

The template provided sets up two vectors `x` and `y` with some
initial values. The exercise is to complete the `ddot()` kernel
which we will give the prototype:
```
  __global__ void ddot(int n, double * x, double * y, double * result);
```
where the `result` is a single scalar value which is the dot
product. A naive serial kernel is provided to give the correct
result.

Suggested procedure
1. Use a `__shared__` temporary variable to store the contribution from
each different thread in a block, and then compute the sum for the block.
2. Accumulate the sum from each block to the final answer.

Remember to deal correctly with any array 'tail'.

Some care may be needed to check the results for this problem. For
debugging, one may want to reduce the problem size; however, there
is the chance that an erroneous code actually gives the expected
answer by chance. Be sure to check with a larger problem size.

### Finished?

It is possible to use solely `atomicAdd()` to form the result (and not
do anything using `__shared__` within a block)? Investigate the performance
implications of this (particularly, if the problem size becomes larger).
You will need two versions of the kernel.
