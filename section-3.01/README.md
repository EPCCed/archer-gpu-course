# Exercise



## Matrix-vector product

We should now be in a position to be able to implement a BLAS-like
matrix vector product:
```
  y_i = alpha A_ij x_j + beta y_i
```
where `A_ij` is a matrix with `m` rows and `n` columns, while `x` is
a vector of length `n` and y is a vector of length `m`. Both `alpha`
and `beta` are scalar constants.

We will assume beta = 0, so that we have the slightly more simple
update
```
  y_i = alpha A_ij x_j
```

### Exercise

Write a code to compute a matrix vector product.

Suggested procedure

1. Make the simplifiying assumption that 1 block per row will be
   used, and that the number of columns is equal to
   the number of threads per block. The number of blocks in the
   x-direction (rows) should be `m` while the number of blocks in
   the y-direction should be 1. This should allow
   elimination of the loop over rows and the loop over columns.

2. The limitation of one block per row must be relaxed. Gerenalise
   the kernel (and launch parameters) so that a two-dimension grid
   of blocks can be used.
3. Finally, some care may be required to ensure the coalescing
   behaviour is favourable.
