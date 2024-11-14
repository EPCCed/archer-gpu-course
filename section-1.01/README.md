# Introduction

- A simple model of performance
- Graphics processors

## A simple model of performance

A very simple picture of a computer might be

![A simple picture of CPU/memory](../images/ks-schematic-simple.svg)

Correspondingly, there might be a number of factors which could be
taken into consideration in a performance model:

1. Clock speed: the rate of issue of instructions by the processor
2. Memory latency: time taken to retrieve a data item from memory
3. Memory bandwidth: amount of data transferred in unit time
4. Parallelism: can I replicate the basic unit above?


### Clocks speeds

Processor clock speed determines the fundamental rate of processing
instructions, and hence data.

Historically, increases in CPU performance have been related to increases
in clock speed. However, owing largely to power constraints, most modern
processors have a clock speed of around 2-3 GHz.

Absent some unforeseen fundamental breakthrough, it is not expected that
this fundamental speed will increase significantly in the future.

### Memory latency

Memory latency is a serious concern. It may take O(100-1000)
clock cycles, from a standing start, to retrieve a piece of
data from memory to the processor (where it can be held and
operated on in a register).


CPUs mitigate this problem by having caches: memory that is
"closer" to the processor, and so reduces the time for access.
Many caches are hierarchical in nature: the nearer the processor
the smaller the cache size in bytes, but the faster the access.
These are typically referred to  as Level 1, Level 2, Level 3,
(L1, L2, L3) and so on

#### Exercise (1 minute)

Try the command
```
$ lscpu
```
on Cirrus to see what the cache hierarchy looks like.

Other latency hiding measures exist, e.g., out-of-order execution
where instructions are executed based on the availability of data,
rather than the order originally specified.

### Memory bandwidth

CPUs generally have commodity DRAM (dynamic
random access memory). While the overall size of memory can
vary on O(100) GB, the exact size may be a cost (money) decision.
In the same way, the maximum memory bandwidth is usually limited
to O(100) GB/second for commodity hardware.

Note that many real applications are memory bandwidth limited
(the bottleneck is moving data, not performing arithmetic operations).
Memory bandwidth can then be a key consideration.

### Parallelism

While it is not possible to increase the clock speed of an individual
processor, one can use add more processing units (for which we will
read: "cores").

Many CPUs are now mutli-core or many-core, with perhaps O(10) or O(100)
cores. Applications wishing to take advantage of such architectures
*must* be parallel.

#### Exercise (1 minute)

Look at `lscpu` again to check how many cores, or processors, are
available on Cirrus.


## Graphics processors

Driven by commercial interest (games), a many-core processor *par exellence*
has been developed. These are graphics processors. Subject to the same
considerations as those discussed above, the hardware design choices taken
to resolve them have been specifically related to the parallel pixel
rendering problem (a trivially parallel problem).

Clocks speeds have, historically, lagged behind CPUs, but are now
broadly similar. However, increases in GPU performance are related to
parallelism.

Memory latency has not gone away, but the mechanism used to mitigate it
is to allow very fast switching between parallel tasks. This eliminates
idle time.

GPUs typically have included high-bandwidth memory (HBM): a relatively
small capacity O(10) GB but relatively large bandwidth O(1000) GB/second
configuration. This gives a better balance between data supply and data
processing for the parallel rendering problem.

So GPUs have been specifically designed to solve the parallel problem of
the rendering of independent pixels. A modern GPU may have O(1000) cores.


### Hardware organiastion

Cores on NVIDIA GPUs are organised into units referred to as
*streaming multiprocessors*, or SMs. There might be 32 or
64 cores per SM, e.g., depending on 32 or 64 bit operations.
More recent architectures also have "tensor" cores for 16-bit
(half precision) operations.

Each SM has its own resources in terms of data/instruction caches,
registers, and floating point units.
The Cirrus V100 GPU cards have 80 SMs each (so 80x32 = 2560 cores
for double precision arithmetic).


A more complete overview is given by NVIDIA
https://developer.nvidia.com/blog/inside-volta/

For AMD GPUs, the picture is essentially similar, although some of the
jargon differs.


## Host/device (historical) picture

GPUs are typically 'hosted' by a standard CPU, which is responsible
for orchestration of GPU activities. In this context, the CPU and GPU
are often referred to as *host* and *device*, respectively.

![Host/device schematic](../images/ks-schematic-host-device.svg)

There is clearly potential for a bottleneck in transfer of data
between host and device.


A modern configuration may see the host (a multi-core CPU) host 4-8
GPU devices.


## Host/device picture

The most recent hardware has attempted to address the potential
bottleneck in host/dvice transfer by using a higher bandwidth
"chip-to-chip " connection.

![Host/device schematic](../images/ks-schematic-host-device-recent.svg)

This model here is typically 1 CPU associated with 1 GPU.
