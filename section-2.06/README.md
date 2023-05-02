# Constant memory

One further type of memory used in CUDA programs is *constant* memory,
which is device memory to hold values which cannot be updated for
the duration of a kernel.

Physically, this is likely to be a small cache on each SM set aside for
the purpose.

This can provide fast (read-only) access to frequently used values.
It is a limited resource (exact capacity may depend on particular
hardware).

## Kernel parameters

If one calls a kernel function, actual arguments are (conceptually, at
least) passed by value as in standard C, and are placed in constant memory.
E.g.,
```
__global__ void kernel(double arg1, double * arg2, ...);
```
If one uses the `--ptxas-options=-v` option to `nvcc` this will
report (amongst other things) a number of `cmem[]` entries;
`cmem[0]` will usually include the kernel arguments.

Note this may be of limited size (e.g., 4096 bytes), so large
objects should not be passed by value to the device.

### Static

It is also possible to use the `__constant__` memory space qualifier
for objects declared at file scope, e.g.:
```
static __constant__ double data_read_only[3];
```
Host values can be copied to the device with the API function
```
  double values[3] = {1.0, 2.0, 3.0};

  cudaMamcpyToSymbol(data_read_only, values, 3*sizeof(double));
```
The object `data_read_only` may then be accessed by a kernel or kernels
at the same scope.

The compiler usually reports usage under `cmem[3]`. Again, capacity
is limited (e.g., may be 64 kB). If an object is too large it will
probably spill into global memory.

## Exercise

We should now be in a position to combine our matrix operation, and
the reduction required for the vector product to perform another
useful operation: a matrix-vector product. For a matrix `A_mn` of
`m` rows and `n` columns, the product `y_i = alpha A_ij x_j` may be
formed with a vector `x` of length `n` to give a result `y` of
length `m`. `alpha` is a constant.

A new template has been provided. A simple serial version has been
implemented with some test values.

Suggested procedure:
1. To start, make the simplifying assumption that we have only 1 block
   per row, and that the number of columns is equal to the number of
   threads per block. This should allow the elimination of the loop
   over both rows and columns with judicious use of thread indices.
2. The limitation to one block per row will harm occupancy. So we
   need to generalise to allow columns to be distributed between
   different blocks. Hint: you will probably need a two-dimensional
   `__shared__` provision in the kernel. Use the same total number
   of threads per block with `blockDim.x == blockDim.y`.
3. Leave the concern of coalescing until last. The indexing can be rather
   confusing.

### Finished

A fully robust solution might check the result with a rectangular
thread block.

A level 2 BLAS implementation may want to compute the update
` y_i := alpha A_ij x_j + beta y_i`. How does this complicate
the simple matrix-vector update?
