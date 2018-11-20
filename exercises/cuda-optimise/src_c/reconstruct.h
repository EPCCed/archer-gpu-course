/*
 * This is a CUDA code that performs an iterative reverse edge 
 * detection algorithm.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2013 
 */


/* dimension of the image is NxX */
#define N 2048

/* Number of iterations to run */
#define ITERATIONS 100


/* The actual CUDA kernel that runs on the GPU - 1D version by column */
__global__ void inverseEdgeDetect(float d_output[N+2][N+2],
                                  float d_input[N+2][N+2],
		float  edge[N+2][N+2]);
/* The actual CUDA kernel that runs on the GPU - 2D version */
__global__ void inverseEdgeDetect2D(float d_output[N+2][N+2], float d_input[N+2][N+2], float d_edge[N+2][N+2]);


/* Forward Declarations of utility functions*/
double get_current_time();
void datread(char*, void*, int, int);
void pgmwrite(char*, void*, int, int);
void checkCUDAError(const char*);
