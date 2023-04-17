
# Introduction to GPU programming with CUDA/HIP

This short course will provide an introduction to GPU computing with CUDA
aimed at scientific application programmers wishing to develop their own
software. The course will give a background on the difference between CPU
and GPU architectures as a prelude to introductory exercises in CUDA
programming. The course will discuss the execution of kernels, memory
management, and shared memory operations. Common performance issues are
discussed and their solution addressed. Profiling will be introduced via
the current NVIDIA tools.

The course will go on to consider execution of independent streams, and
the execution of work composed as a collection of dependent tasks expressed
as a graph. Device management and details of device to device data transfer
will be covered for situations where more than one GPU device is available.
CUDA-aware MPI will be covered.

The course will not discuss programming with compiler directives, but does
provide a concrete basis of understanding of the underlying principles of
the CUDA model which is useful for programmers ultimately wishing to make
use of OpenMP or OpenACC (or indeed other models). The course will not
consider graphics programming, nor will it consider machine learning
packages.

Note that the course is also appropriate for those wishing to use AMD GPUs
via the HIP API, although we will not specifically use HIP.

Attendees must be able to program in C or C++ (course examples and
exercises will limit themselves to C). A familiarity with threaded
programming models would be useful, but no previous knowledge of GPU
programming is required.

## Installation


## Timetable

The timetable may shift slightly in terms of content, but we will stick to
the advertised start and finish times, and the break times.

### Day one

| Time  | Content                                  | Section       |
|-------|------------------------------------------|---------------|
| 09:30 | Logistics, login, modules, local details | See above     |
| 10:00 | Introduction                             |               |
|       | Performance model; Graphics processors   | [section-1.01](section-1.01)    |
| 10:30 | The CUDA/HIP programming model           |               |
|       | Host/device model; memory, kernels       | [section-1.02](section-1.02)      |
| 11:00 | Break                                    |               |
| 11:30 | CUDA/HIP programming                     | []()          |
|       | Memory management, exercise              | section-2.01  |



### Day two
