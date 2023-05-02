# Conjugate gradient

This is a slightly more substantial programming problem.


## Conjugate gradient solver

The conjugate gradient method provides a general
method for the solution of linear systems _Ax = b_ for symmetric
matrices. The algorithm is explained here:

https://en.wikipedia.org/wiki/Conjugate_gradient_method

It is not necessary to understand the details, as we are just
interested in implementing the algorithm.

There are two main steps involved. The first is to perform a
matrix-vector multiplication. This was an earlier exercise
for which a solution is provided in the current directory.

The second step is to compute a scalar residual from a vector
residual. If we have a vector `r` of length `n` this can be
done in serial with:
```
  residual = 0.0;
  for (int i = 0; i < n; i++) {
    residual += r[i]*r[i];
  }
```
This you should also recognise as the solution to an earlier
exercise (a version is also provided).

The exercise then means using these parts to construct the
entire algorithm. Restrict yourself to the use of a single
GPU.

Remember that you can always check your answer by multiplying
out the solution to see if you recover the original right-hand
side. This can be done on the host.
