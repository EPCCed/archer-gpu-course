__kernel void negate(__global int *d_a)
{
    int idx = get_global_id(0);
    d_a[idx] = -1 * d_a[idx];
}
