KOKKOS_PATH = ${KOKKOS_DIR}/CudaUVM
KOKKOS_CUDA_OPTIONS=force_uvm ,enable_lambda
EXE = ${EXE_NAME}.CudaUVM
include 02_Exercise.mk

