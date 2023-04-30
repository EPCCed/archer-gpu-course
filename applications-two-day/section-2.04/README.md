# Managed memory

CUDA provides a number of different ways to establish device
memory and transfer memory between host and device.


Different mechanisms may be favoured in different situations.


## Explicit memory allocation/copy

We have seen the explicit mechanics using standa C pointers.
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
considering the transfers of large contigous blocks of data.

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

This established an effective single reference to memory which can be
accessed on both host and device.

Host/device transfers are managed automatically as the need arises.

So, a schematic of usage might be:
```

  double * ptr = NULL;

  cudaMallocManaged(&pre, nbytes);

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

1. `cudamemAdviseSetReadMostly` indicates infrequent reads;
2. `cudaMemAdviseSetPreferredLocation` sets the preferred location to
   the specified device (`cudaCpuDeviceId` for the host); 
3. `cudaMemAdviseSetAccessedBy` suggests that the data will be accessed
   by the specfied device.

Each option has a corresponding `Unset` value which can be used to
nullify the effect of a preceding `Set` specification.

Again, the relevant memory must be managed by CUDA.



## When to use

Often useful to start development and testing with managed memory, and
then move to explicit `cudaMalloc()/cudaMmecpy()` if it is required for
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

Add the relevant prefetches.

