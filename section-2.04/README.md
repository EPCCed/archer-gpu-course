# Managed memory

CUDA provides a number of different ways to establish device
memory and transfer data between host and device.


Different mechanisms may be favoured in different situations.


## Explicit memory allocation/copy

We have seen the explicit mechanics of using standard C pointers.
Schematically:
```

  double * h_ptr = NULL;
  double * d_ptr = NULL;

  h_ptr = (double *) malloc(nbytes);

  cudaMalloc(&d_ptr, nbytes);
  cudaMemcpy(d_ptr, h_ptr, nbytes, cudaMemcpyHostToDevice);
```
The host pointer to the device memory is then used in the kernel invocation.
```
  myKernel<<<...>>>(d_ptr);
```
However, pointers to device memory cannot be dereferenced on the host.

This is a perfectly sound mechanism, particularly if we are only
considering the transfers of large contiguous blocks of data.
(It is also likely to be the fastest mechanism.)

However, this can become onerous if there are complex data access
patterns, or if rapid testing and development are required. It also
gives rise to the need to have both a host reference and a device
reference in the code (`h_ptr` and `d_ptr`).


## Managed memory (or unified memory)

Managed memory is allocated on the host via
```
__host__ cudaErr_t cudaMallocManaged(void ** ptr, size_t sz, ...);
```
in place of the combination of `malloc()` and `cudaMalloc()`.

This establishes an effective single reference to memory which can be
accessed on both host and device.

Host/device transfers are managed automatically as the need arises.

So, a schematic of usage might be:
```

  double * ptr = NULL;

  cudaMallocManaged(&ptr, nbytes);

  /* Initialise values on host ... */

  for (int i = 0; i < ndata; i++) {
    ptr[i] = 1.0;
  }

  /* Use data in a kernel ... */
  kernel<<<...>>>(ptr);
```

### Releasing managed memory

Managed memory established with `cudaMallocManaged()` is released with
```
  cudaFree(ptr);
```
which is the same as for memory allocated via `cudaMalloc()`.


### Mechanism: page migration

Transfers are implemented through the process of page migration.
A page is the smallest unit of memory management and is often
4096 bytes on a typical (CPU) machine. For CUDA managed memory
the page size is often 64K bytes.

Assume - and this may or may not be the case - that
`cudaMallocManaged()` establishes memory in the host space.
We can initialise memory on the host and call a kernel.

When the GPU starts executing the kernel, any access to the
relevant (virtual) address is not present on the GPU, and
the GPU will issue a page fault.

The relevant page of memory must then be migrated (i.e., copied)
from the host to the GPU before useful execution can continue.

Likewise, if the same data is required by the host after the kernel,
an access on the host will trigger a page fault on the CPU, and the
relevant data must be copied back from the GPU to the host.

### Prefetching

If the programmer knows in advance that memory is required on the
device before kernel execution, a prefetch to the destination
device may be issued. Schematically:
```

  cudaGetDevice(&device);
  cudaMallocManaged(&ptr, nbytes);

  /* ... initialise data ... */

  cudaMemPrefetchAsync(ptr, nbytes, device);

  /* ... kernel activity ... */
```
As the name suggests, this is an asynchronous call (it is likely to return
before any data transfer has actually occurred).
It can be viewed as a request to the CUDA run-time to transfer the
data.

The memory must be managed by CUDA.

Prefetches from the device to the host can be requested by using the special
destination value `cudaCpuDeviceId`.


### Providing hints

Another mechanism to help the CUDA run-time is to provide "advice".
This is done via
```
  __host__ cudaErr_t cudaMemAdvise(const void * ptr, size_t sz,
                                   cudaMemoryAdvise advice, int device);
```
The `cudaMemoryAdvise` value may include:

1. `cudaMemAdviseSetReadMostly` indicates infrequent reads;
2. `cudaMemAdviseSetPreferredLocation` sets the preferred location to
   the specified device (`cudaCpuDeviceId` for the host);
3. `cudaMemAdviseSetAccessedBy` suggests that the data will be accessed
   by the specified device.

Each option has a corresponding `Unset` value which can be used to
nullify the effect of a preceding `Set` specification.

Again, the relevant memory must be managed by CUDA.



## When to use

Often useful to start development and testing with managed memory, and
then move to explicit `cudaMalloc()/cudaMemcpy()` if it is required for
performance and is simple to do so.


## Exercise (15 minutes)

It the current directory we have supplied as a template the solution
to the exercise to the previous section. This just computes the
operation `A_ij := A_ij + alpha x_i y_j`.

It may be useful to run the unaltered code once to have a reference
`nvprof` output to show the times for different parts of the code.
`nvprof` is used in the submission script provided.

Confirm you can replace the explicit memory management using
`malloc()/cudaMalloc()` and `cudaMemcpy()` with managed memory.
It is suggested that, e.g., both `d_a` and `h_a` are replaced
by the single declaration `a` in the main function.

Run the new code to check the answers are correct, and the new output
of `nvprof` associated with managed (unified) memory.


Add the relevant prefetch requests for the vectors `x` and `y` before
the kernel, and the matrix `a` after the kernel. Note that the device
id is already present in the code as `deviceNum`.


### What can go wrong?

What happens if you should accidentally use `cudaMalloc()` where you intended
to use `cudaMallocManaged()`?
