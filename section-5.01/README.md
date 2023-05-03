# Device Management

## Available devices

The number of GPUs currently available to the host can be obtained
via the API call
```
  int ndevice = -1;
  cudaGetDeviceCount(&ndevice);
```
Note that the total number visible may be controlled ultimately
by the scheduling system, or some other external consideration.

Devices are numbered logically `0,1,2, ..., ndevice-1`. The identity
of the currently 'set' device, or current context, is
```
  int myid = -1;
  cudaGetDevice(&myid);
```
This will be `0` if there is one device. All CUDA API calls, and
kernel launches then involve this device.

## More then one device

If we have more than one GPU available to the host process, then
`cudaGetDeviceCount()` will return the appropriate number. The
initial context will still be device `0`, the default.

We can make use of the other devices by switching context with,
e.g.,
```
  int myid1 = 1;
  cudaSetDevice(myid1);
```
An API call will then refer to the new device. E.g.,
```
  double * d_data1 = NULL;
  cudaMalloc(&d_data1, ndata*sizeof(double));
```
will allocate memory on the current device.

Managed memory is slightly different. The CUDA runtime will keep track
of what is required where.

The same is true for kernels: a kernel is launched on the current
device.

### Management



## Peer Access

If one has two memory allocations on the same GPU it is perfectly
valid to do:
```
  cudaMemcpy(d_ptr1, d_ptr2, sz, cudaMemcpyDeviceToDevice);
```
which is a copy within device memory.

More recent NVIDIA (and AMD) devices provide additional fast links between
GPUs within a node. (NVIDIA refer to this as NVLINK.) These bypass the
need to transfer data via the host (or run a kernel).

PICTURE

This is referred to as "peer access" in the CUDA API.

### Querying capability

In general, one should ensure peer access via:
```
  cudaDeviceCanAccessPeer(int * canAccessPeer, int device1, int device2);
```
where `device1` is the destination device, and `device2` is the source
device.

If available, it is possible to disable and enable the peer access using
```
  cudaDeviceDisablePeerAccess(int peerDevice);
  cudaDeviceEnablePeerAccess(int perrDevice, unsigned int flags);
```
(`flags` is always set to zero).

When possible, an enabled the relevant link should be used for
`cudaMemcpyDeviceToDevice` copies.

## Exercise

Write a simple program which allocates a large array (at least 10
MB, say) on each of two devices using `cudaMalloc()` (the same size
on each device). By making repeated copies of the array with
`cudaMemcpy()`, try to assess the bandwidth which can be obtained by

1. coping from host to device, and then from device to host;
2. copying directly from one device to another using `cudaMemcpyDeviceToDevice`
   with peer access *disabled*;
3. repeating with peer access enabled.

Note that we will need to adjust our queue submission script to ensure
that two GPUs are available to the program.
