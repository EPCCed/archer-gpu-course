""" This is a simple pycuda example that multiplies each element
    of an array by a constant.
    It introduces the concepts of device memory management, and
    kernel invocation.

    Training material developed by Kevin Stratford (kevin@epcc.ed.ac.uk)
    Copyright EPCC, The University of Edinburgh 2022 """

# Provide access to the driver, and the kernel compiler
# Use numpy arrays

import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Define the problem size, and the number of blocks/threads

int ARRAY_SIZE        = 256
int NUM_BLOCKS        = 1
int THREADS_PER_BLOCK = 256

# Define the kernel here
# On exit, we require each element of x[] by replaced by ax[]. 

kernel_code = SourceModule("""
  __global__ void scale_vector(double a, double * x) {

  }
""")

# Host code
# Establish some values on the host
# Also establist results array ax

x = numpy.ones(ARRAY_SIZE, dtype = double)
a = 2.0

ax = numpy.zeros(ARRAY_SIZE, dtype = double)

# Allocate memory on the device
# and copy initial data to device

x_d = cuda.mem_alloc(x.nbytes)

cuda_memcpy_htod(x_d, x)

# Obtain the kernel function
# This may generate compilation errors

blocks = (NUM_BLOCKS, 1, 1)
threads_per_block = (THREADS_PER_BLOCK, 1, 1)

kernel = kernel_code.get_function("scale_vector")

kernel(a, x_d, blocks, threads_per_block)

# Copy back the results, and check

cuda.memcpy_dtoh(ax, x_d)

check = numpy.allclose(a*x, ax, atol = numpy.finfo(double).eps)

if (check):
    print("Values are correct")
else:
    print("Values are not correct")
