# Introduction

- A simple model of performance
- Graphics processors

## A simple model of performance

A very simple picture of a computer might be
```
  MEMORY    ->    PROCESSOR
```

1. Clock speed: the rate of issue of instructions
2. Memory latency: time taken to retreive a datum from memory
3. Memory bandwidth: amount of data transferred in unit time
4. Parallelism: can I replicate the basic unit above?

Clock speeds have largely ceased to increase, owing to power
considerations. Most modern processors have a clock speed of
around 2 GHz. Absent some unforseen fundamental breakthrough,
it is not expected that this will increase signficantly.

Memory latency is a serious concern. It may take O(100-1000)
clock cycles to retrieve a piece of data from memory to the
processor (where it is held in a register).


CPUs mitigate this problem by having caches: memory that is
"closer" to the processor, and so reduced the time for access.
Many caches are heirarchical in nature: the nearer the processor
the smaller the cache size in bytes, but the faster the access.
These are typically referred to  as Level 1, Level 2, Level 3,
(L1, L2, L3) and so on (often available for inspection on
Unix-like systems via commands such as `lscpu`).

Other latency hiding measures exist, e.g., out-of-order execution
where instructions are executed based on the availability of data,
rather than the order originally specified.


Memory bandwidth. CPUs generally have commodity DRAM (dynamic
random access memory). While the overall size of memory can
vary on O(100) GB, the exact size may be a cost (momey) decision.
In the same way, the maximum memory bandwidth is usually limited
to O(100) GB/second for commodity hardware.
Note that many real applications are memory bandwidth limited
(the bottleneck is moving data, not performing arithmetic operations).


Parallelism. While it is not possible to increaes the clock speed
of an indivdual processor, one can use add more processing units
(for which we will read: "cores").


## Graphics processors

Driven by commercial interest (games), a many-core processor *par exellence*
has been developed. These are subject to the same sort of considerations
as thoese discussed above, but the hardware design choices taken to resolve
them have been specificially related to the parallel pixel rendering
problem.

Clocks speeds are similar.

Memory latency has not gone away, but the mechanism used to mitigate it
is to allow very fast switching between parallel tasks.

GPUs typically have included high-bandwidth memory (HBM): a relatively
small capacity O(10) GB but relatively large bandwidth O(1000) GB/second
configuration. This gives a better balanced hardware for the parallel
rendering problem.




## Host/device picture

GPUs are typically 'hosted' by a traditional CPU, which is responsible
for orchestration of GPU activities. In this context, the CPU and GPU
are often referred to as *host* and *device*, respectively.

The first important point is that the two have distinct memories
(hardware) and distinct memory address spaces (software).

