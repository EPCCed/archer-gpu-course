# Profiling

So far we have used the simple text-based report `nvprof`. NVIDIA now
provide a much more sophisticated suite of tools for profiling of
GPU codes. There are two elements. First, Nsight Systems is

https://developer.nvidia.com/nsight-systems

Second, Nsight Compute deals with kernels.

https://developer.nvidia.com/nsight-compute


The first port-of-call should always be Nsight Systems.


## Nisght Systems

To use, compile and run as usual, and prefix the executable
```
nsys profile -o systems ./a.out
```
which should produce a file `systems.nsys-rep`. This can be
read into the user interface Nsight Systems.

The usual mode of operation is to copy the report file back to
your local machine.

### Adding NVTX markers

A bare profile shows CUDA API (host) activity and device activity
(memory copies and kernels). It can be useful to identify particular
sections of code.

NVTX (NVIDIA Toolkit extension) markers can be added to code (host
code) as follows. Include the file
```
#include "nvToolsExt.h"
```
Identify the region of code of interest, and add a range start and
end with, e.g.,
```
  nvtxRangeId_t id = nvtxRangeStartA("MY ASCII LABEL");

  /* ... code of interest ... */

  nvtxRangeEnd(id);
```
and recompile (you may needto add `-lnvToolsExt`).

This will cause a coloured bar to appear in the profile indicating
the relevant duration.

## Nsight compute

If one has concerns about the performance of a particular kernel, or
kernels, then one needs to turn to Nsight Compute.

For basic information use
```
ncu -o default ./a.out
```
which should produce a file `default.ncu-rep`. This report file is
loaded into Nsight Compute.

For more detailed information one can use, e.g.,
```
ncu --set detailed -o detailed ./a.out
```
and for full information use
```
ncu --set full -o full ./a.out
```
which will run additional passes of the kernel to collect more
metrics.

Note that running `ncu` can be quite time-consuming, and if using
a real application, a small problem size should be selected in the
first instance. One can also use filters to limit the information
collected (e.g., for an individual kernel).


## Exercise

For the simple matrix operation we developed earlier, try to run
first Nsight Systems, and then Nsight Compute with the various
options.

Have a go at adding some NVTX markers to highlight a region of host code.
