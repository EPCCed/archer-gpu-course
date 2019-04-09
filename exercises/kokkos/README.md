
# Introduction to Kokkos

This series of exercises will introduce the Kokkos framework for portable
performance using both OpenMP and CUDA as a backend.

The examples follow the first few of a longer sequence of tutorial examples
appearing in the SANDIA Kokkos tutorial found at 
https://github.com/kokkos/kokkos-tutorials

The full Kokkos programming guide can be found at https://github.com/kokkos/kokkos/wiki

Instructions are provided below (and within the templates) to allow you to
complete each of the exercises.

## (1) Initialise, Use, Finalise

The first exercise asks you to initialise and finalise Kokkos, and use the
patterns `parallel_for` and `parallel_reduce` in the computation of an
inner product.

The corresponding template source code is in subdirectory `1`; ie., 
1/exercise_1_begin.cpp`. The places where you need to change things
are marked in the code with "EXERCISE".

You will need to recall the patterns:
```cpp
  Kokkos::parallel_for(N, KOKKOS_LAMBDA (int i) {
    //... loop body with index i of extent N
  });
```
where the macro `KOKKOS_LAMBDA` is representing the capture. The
`parallel_reduce` pattern is of the form
```cpp
  Kokkos::parallel_reduce(M, KOKKOS_LAMBDA (int j, double & sum) {
    // sum += ...loop body with index j of extent M
  }, result);
```
where `result` is a variable defined in the outer scope to hold the result.
The `sum` variable is managed by Kokkos.


You can compile the code using the OpenMP backend with the command:


```shell
$ make OpenMP
```

When you run you can set the problem size with command line flags.
These are passed as a power of 2 (e.g. `10 => 2**10 = 1024`).
* `-N` - the number of rows (default = 12)
* `-M` - the number of columns (default = 10)
* `-S` - the total size (default = 22, must equal sum of M and N)

Can also specify the number of repeats:
* `-nrepeat` (default = 100)

When using OpenMP, the number of threads to use by setting the
`OMP_NUM_THREADS` environment variable. This is done for a
range of values in the supplied script.


```shell
$ qsub submit.sh
```

## (2) Use Views

The goal of this exercise will be to replace the raw memory allocations
with Kokkos Views, and corrresponding memory accesses
with the relevant Kokkos view access.

The source code is found in `2/exercise_2_begin.cpp`.
Again, code requiring attention is marked with 'EXERCISE'.

Recall that a View may be declared via, e.g.,
```cpp
Kokkos::View < double * > x("my vector", nElements);
```
and access to individual elements is via brackets, e.g.,
```cpp
x(index) = ...
```

Note we force Kokkos use to UVM for the CUDA build.


```shell
$ make OpenMP
$ make CudaUVM
```

Both the OpenMP and Cuda version are run by the script:

```shell
$ submit submit.sh
```


## (3) Use Mirror Views

Now, we will replace use of managed GPU memory with explicit
data management via Kokkos mirror views and and copies.

The exercise template is `3/exercise_3_begin.cpp`

Recall the copy is
```cpp
Kokkos::deep_copy(dest, src);
```
for the required direction of transfer.

Two versions may be compiled.

```shell
$ make OpenMP
$ make Cuda
```

The supplied script runs both versions

```shell
$ qsub submit.sh
```


## (4) Control the Layout

The final exercise provides some wide-ranging options to investigate memory
layouts. memory and execution spaces; using a
Kokkos RangePolicy to parallelise the inner loop.

The template is `4/exercise_4_begin.cpp` contains further instructions and
hints.



```shell
$ make
```
