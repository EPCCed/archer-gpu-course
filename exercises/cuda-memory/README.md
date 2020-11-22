
# Using constant and shared memory

In this exercise we will look at some simple operations using CUDA constant
memory and shared memory. A working program (in both C and Fortran) is
supplied which uses a kernel to reverse the elements stored in an array.
The exercise will make some changes to this program.

The exercise was created by EPCC, The University of Edinburgh. Documentation
and source code copyright The University of Edinburgh 2019.

Contributors: Alan Gray, Kevin Stratford


## Introduction

Where necessary, you can refer to the CUDA C Programming Guide and
Reference Manual documents available from
http://developer.nvidia.com/nvidia-gpu-computing-documentation
 


## 1) Using constant memory

Have a look at the code to see what it does. The kernel <code>reverse()</code>
takes the elements of an array $a_1, a_2, a_3, \ldots$ and reverses their
order so that the output appears $\ldots, a_3, a_2, a_1$. You should see
that the size of the array involved in the reverse operation is set at
compile time. (This is done in different ways in C and Fortran.)

Choose the C or Fortran version from the appropriate subdirectory.


### C Version:

Suppose we want to set the size of the array at run time. If this were the
case, it might be appropriate to replace the <code>ARRAY_SIZE</code>
definition in the kernel with a variable declared in constant memory.
(One could also pass the size as an actual argument of the kernel routine.)

As an exercise, replace the <code>ARRAY_SIZE</code> definition with a
variable with a <code> \__constant__</code> declaration. Keep the same
value of the array size to start with, and remember to initialise the
value before the kernel invocation. Check the program still reports
the correct answer.


### Fortran:

Suppose we want to set the array size at run time, instead of using the
<code>parameter</code> definition in the main program (and removing the
actual argument to the kernel routine). It would be appropriate to do
this using a variable declared with the <code>constant</code> attribute.
Where should this variable be declared so that is available in both the
kernel and the main program?

Assign the appropriate value before the kernel is invoked. Check the
program still reports the correct answer.


## 2) Using Shared Memory

### C and Fortran:

We will now use CUDA shared memory to produce a new version of the
<code>reverse()</code> where each block uses a temporary variable to store
a its contributions to the output array. While this is not strictly
necessary, it provides some practice without introducing too much complexity.

1) In the kernel, declare an array of the right length so that each thread in
   a block can store one value of the integer data type in shared memory.

2) _Within each block_, store the relevant elements of the input array in
   the temporary array, but with order reversed.

3) Copy the reversed elements from the temporary array to the output array.
   You will have to compute the correct offset for each block in the global
   problem size to do this.

4) Recompile the code, and check it still reports the correct answer.


### Compilation and Execution

In both C and Fortran versions, compilation is via

```shell
$ make
```

while submission to be queue system is via
```shell
$ sbatch submit.sh
```


### Keeping a copy of your work

The course materials and the exercise templates are available via the course
repoistory (which includes stock solutions).

If you wish to keep a copy of your particular solution, please
make a copy to a suitable location.

