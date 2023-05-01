
<img src="./images/archer2_logo.png" align="left" width="355" height="100" />
<img src="./images/epcc_logo.jpg" align="right" width="133" height="100" />

<br><br><br><br>

# Introduction to GPU programming with CUDA/HIP

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

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

For details of how to log into a Cirrus account, see
https://cirrus.readthedocs.io/en/main/user-guide/connecting.html

Check out the git repository to your Cirrus account.
```
$ cd ${HOME/home/work}
$ https://github.com/EPCCed/archer-gpu-course.git
$ cd archer-gpu-course
```
For the examples and exercises in the course, we will use the
NVIDIA compiler driver `nvcc`. To access this
```
$ module load nvidia/nvhpc
```
Check you can compile and run a very simple program
and submit the associated script to the queue system.
```
$ cd section-2.01
$ nvcc -arch=sm_70 exercise_dscal.cu
$ sbatch submit.sh
```
The result should appear in a file `slurm-123456.out` in the working
directory.

Each section of the course is associated with a different directory, each
of which contains a number of example programs and exercise templates.
Answers to exercises generally re-appear as templates to later exercises.
Miscellaneous solutions also appear in the solutions directory.


## Timetable

The timetable may shift slightly in terms of content, but we will stick to
the advertised start and finish times, and the break times.

### Day one

| Time  | Content                                  | Section                      |
|-------|------------------------------------------|------------------------------|
| 09:30 | Logistics, login, modules, local details | See above                    |
| 10:00 | Introduction                             |                              |
|       | Performance model; Graphics processors   | [section-1.01](section-1.01) |
| 10:30 | The CUDA/HIP programming model           |                              |
|       | Abstraction; host code and device code   | [section-1.02](section-1.02) |
| 11:00 | Break                                    |                              |
| 11:30 | CUDA/HIP programming                     |                              |
|       | Memory management, exercise              | [section-2.01](section-2.01) |
| 12:15 | CUDA/HIP programming (cont.)             |                              |
|       | Kernels, exercise                        | [section-2.02](section-2.02) |
| 13:00 | Lunch                                    |                              |
| 14:00 | Some performance considerations          |                              |
|       | Exercise on matrix operation             | [section-2.03](section-2.03) |
| 15:00 | Break                                    |                              |
| 15:20 | More on memory: managed memory           |                              |
|       | Exercise on managed memory               | [section-2.04](section-2.04) |
| 15:50 | More on memory: shared memory            |                              |
| 16:10 | Exercise on vector product               | [section-2.05](section-2.05) |
| 16:30 | All together: matrix-vector product      | [][]                         |
| 17:00 | Close                                    |                              |


### Day two


| Time  | Content                                  | Section                      |
|-------|------------------------------------------|------------------------------|
| 09:00 | Detour: visual profiler                  |                              |
| 09:10 | Exercise: nsight systems and compute     | [section-3.01](section-3.01)      |
| 09:30 | Streams                                  |                              |
| 09:50 | Streams exercise                         | [section-4.01](section-4.01) |



---
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]
