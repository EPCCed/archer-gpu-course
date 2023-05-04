# Using MPI with GPUs

If one has more than one GPU, and certainly if one has
more than one node with GPUs, it is natural to ask how
to think about programming with MPI.

First, this may require a design decision about how
to go about the problem.

## Setting the device based on rank

A natural choice may be to run one MPI process per device.
For example, on a node with 4 GPUs, we would ask for 4 MPI
processes. Each individual MPI rank would just set the
current device appropriately.
```
  int rank = -1;              /* MPI rank */
  MPI_Comm_rank(comm, &rank);

  cudaSetDevice(rank % ndevicePerNode);
```
The number of devices per node may be obtained via `cudaGetDeviceCount()`
or it may require external input.

### Passing messages between devices

In order to pass a message between two devices, one might consider:
```
  /* On the sending side ... */
  cudaMemcpy(hmsgs, dmsgs, ndata*sizeof(double), cudaMemcpyDeviceToHost);
  MPI_Isend(hsmgs, ndata, MPI_DOUBLE, dst, ...);

  /* On the receiving side ... */
  MPI_Recv(hmsgr, ndata, MPI_DOUBLE, src, ...);
  cudaMemcpy(dmsgr, hmsgr, ndata*sizeof(), cudaMemcpyHostToDevice);
```
This may very well lead to poor performance.

### GPU-aware MPI

It is possible to use device references in MPI calls on the host.
E.g., the previous example might be replaced by
```
  MPI_Isend(dmsgs, ndata, MPI_DOUBLE, dst, ...);
  MPI_Recv(dsmgr, ndata, MPI_DOUBLE, src, ...)
```
Here `dmsgs` and `dsmgr` are device memory references. If within a node
with fast connections, this should be routed in the appropriate way.
A fall-back to copy via the host may be required for inter-node meaages.

Some architectures have the network interface cards connected directly
to the GPUs (rather than the host), so inter-node transfers there would
also favour use of GPU-aware MPI.

## Exercise

The NVIDIA HPC SDK includes a build of OpenMPI with GPU-aware MPI
enabled. A sample program has been provided with measures the time
taken for messages of different size to be send between to MPI
tasks by the two methods outlined above.

Have a look at the program, and try to compile and run it.
