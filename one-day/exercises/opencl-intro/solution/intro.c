#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <CL/opencl.h>

#include <sys/types.h>
#include <sys/time.h>

void checkOpenCLError(cl_int, char *);
cl_int initCLDevice(cl_device_type devtype, cl_context *ctxt,
                    cl_command_queue *queue);
cl_int getCLKernel(cl_context ctxt, char *filename, char *funcname,
                   cl_kernel *kernel);

/* The number of integer elements in the array */
#define ARRAY_SIZE 256

/*
 * The number of threads per OpenCL work group to use.
 */
#define THREADS_PER_WORK_GROUP 16

/* Main routine */
int main(int argc, char *argv[])
{
    int *h_a, *h_out;
    cl_mem d_a;
    cl_int err;

    cl_context ctxt;
    cl_command_queue queue;
    cl_kernel kernel;

    int i;
    size_t sz = ARRAY_SIZE * sizeof(int);

    /* initialise OpenCL */
    err = initCLDevice(CL_DEVICE_TYPE_GPU, &ctxt, &queue);
    checkOpenCLError(err, "initCLDevice");
    err = getCLKernel(ctxt, "kernel.c", "negate", &kernel);
    checkOpenCLError(err, "getCLKernel");

    /*
     * allocate memory on host
     * h_a holds the input array, h_out holds the result
     */
    h_a = (int *) malloc(sz);
    h_out = (int *) malloc(sz);

    /*
     * allocate memory on device
     */
    d_a = clCreateBuffer(ctxt, CL_MEM_READ_WRITE, sz, NULL, &err);
    checkOpenCLError(err, "buffer allocation");

    /* initialise host arrays */
    for (i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = i;
        h_out[i] = 0;
    }

    /* copy input array from host to GPU */
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, sz, h_a, 0, NULL, NULL);
    checkOpenCLError(err, "buffer write");

    /* run the kernel on the GPU */
    size_t globalsize[1] = { ARRAY_SIZE };
    size_t localsize[1] = { THREADS_PER_WORK_GROUP };

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    checkOpenCLError(err, "setting kernel arguments");

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalsize[0], &localsize[0], 0, NULL, NULL);
    checkOpenCLError(err, "running kernel");

    /* wait for all threads to complete */
    clFinish(queue);

    /* copy the result array back to the host */
    err = clEnqueueReadBuffer(queue, d_a, CL_TRUE, 0, sz, h_out, 0, NULL, NULL);
    checkOpenCLError(err, "buffer read");

    /* print out the result */
    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", h_out[i]);
    }
    printf("\n\n");

    /* free device buffer */
    clReleaseMemObject(d_a);

    /* free host buffers */
    free(h_a);
    free(h_out);

    return 0;
}


cl_int initCLDevice(cl_device_type devtype, cl_context *ctxt, cl_command_queue *queue)
{
    cl_uint num_platforms;
    cl_platform_id *platforms;
    cl_int err;
    int i;
    cl_device_id dev;
    int found;
    
    /* get number of platforms */
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) {
	return err;
    }
    if (num_platforms == 0) {
	return CL_DEVICE_NOT_FOUND;
    }
    /* get IDs for each platform */
    platforms = malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, &num_platforms);
    if (err != CL_SUCCESS) {
	free(platforms);
	return err;
    }

    /* loop over platforms looking for a device of the correct type */
    found = 0;
    for (i = 0; i < num_platforms; i++) {
	err = clGetDeviceIDs(platforms[i], devtype, 1, &dev, NULL);
	if (err == CL_SUCCESS) {
	    found = 1;
	    break;
	}
    }
    free(platforms);
    if (!found) {
	return CL_DEVICE_NOT_FOUND;
    }

    /* now have device ID in dev. create a context and command queue */
    *ctxt = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
	return err;
    }
    *queue = clCreateCommandQueue(*ctxt, dev, 0, &err);
    return err;
}

cl_int getCLKernel(cl_context ctxt, char *filename, char *funcname, cl_kernel *kernel)
{
    cl_int err;
    FILE *f;
    size_t len;
    char *buf;
    cl_program prog;

    /* read the program source file */
    f = fopen(filename, "r");
    if (!f) {
	return CL_INVALID_VALUE;
    }
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);

    buf = malloc(len + 1);
    fread(buf, len, 1, f);
    buf[len] = 0;
    fclose(f);

    /* create program object from source */
    prog = clCreateProgramWithSource(ctxt, 1, (const char **)&buf, NULL, &err);
    free(buf);
    if (err != CL_SUCCESS) {
	return err;
    }

    /* build the program */
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
	return err;
    }

    /* create the kernel object */
    *kernel = clCreateKernel(prog, funcname, &err);
    return err;
}


void checkOpenCLError(cl_int err, char *msg)
{
    char *errstr = "unknown error";
    if (err == CL_SUCCESS) return;
    switch (err) {
    case CL_DEVICE_NOT_FOUND:
	errstr = "device not found";
	break;
    case CL_DEVICE_NOT_AVAILABLE:
	errstr = "device not available";
	break;
    case CL_COMPILER_NOT_AVAILABLE:
	errstr = "compiler not available";
	break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
	errstr = "mem object allocation failure";
	break;
    case CL_OUT_OF_RESOURCES:
	errstr = "out of resources";
	break;
    case CL_OUT_OF_HOST_MEMORY:
	errstr = "out of host memory";
	break;
    case CL_PROFILING_INFO_NOT_AVAILABLE:
	errstr = "profiling info not available";
	break;
    case CL_MEM_COPY_OVERLAP:
	errstr = "mem copy overlap";
	break;
    case CL_IMAGE_FORMAT_MISMATCH:
	errstr = "image format mismatch";
	break;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
	errstr = "image format not supported";
	break;
    case CL_BUILD_PROGRAM_FAILURE:
	errstr = "build program failure";
	break;
    case CL_MAP_FAILURE:
	errstr = "map failure";
	break;
    case CL_INVALID_VALUE:
	errstr = "invalid value";
	break;
    case CL_INVALID_DEVICE_TYPE:
	errstr = "invalid device type";
	break;
    case CL_INVALID_PLATFORM:
	errstr = "invalid platform";
	break;
    case CL_INVALID_DEVICE:
	errstr = "invalid device";
	break;
    case CL_INVALID_CONTEXT:
	errstr = "invalid context";
	break;
    case CL_INVALID_QUEUE_PROPERTIES:
	errstr = "invalid queue properties";
	break;
    case CL_INVALID_COMMAND_QUEUE:
	errstr = "invalid command queue";
	break;
    case CL_INVALID_HOST_PTR:
	errstr = "invalid host pointer";
	break;
    case CL_INVALID_MEM_OBJECT:
	errstr = "invalid mem object";
	break;
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
	errstr = "invalid image format descriptor";
	break;
    case CL_INVALID_IMAGE_SIZE:
	errstr = "invalid image size";
	break;
    case CL_INVALID_SAMPLER:
	errstr = "invalid sampler";
	break;
    case CL_INVALID_BINARY:
	errstr = "invalid binary";
	break;
    case CL_INVALID_BUILD_OPTIONS:
	errstr = "invalid build options";
	break;
    case CL_INVALID_PROGRAM:
	errstr = "invalid program";
	break;
    case CL_INVALID_PROGRAM_EXECUTABLE:
	errstr = "invalid program executable";
	break;
    case CL_INVALID_KERNEL_NAME:
	errstr = "invalid kernel name";
	break;
    case CL_INVALID_KERNEL_DEFINITION:
	errstr = "invalid kernel definition";
	break;
    case CL_INVALID_KERNEL:
	errstr = "invalid kernel";
	break;
    case CL_INVALID_ARG_INDEX:
	errstr = "invalid argument index";
	break;
    case CL_INVALID_ARG_VALUE:
	errstr = "invalid argument value";
	break;
    case CL_INVALID_ARG_SIZE:
	errstr = "invalid argument size";
	break;
    case CL_INVALID_KERNEL_ARGS:
	errstr = "invalid kernel arguments";
	break;
    case CL_INVALID_WORK_DIMENSION:
	errstr = "invalid work dimension";
	break;
    case CL_INVALID_WORK_GROUP_SIZE:
	errstr = "invalid work group size";
	break;
    case CL_INVALID_WORK_ITEM_SIZE:
	errstr = "invalid work item size";
	break;
    case CL_INVALID_GLOBAL_OFFSET:
	errstr = "invalid global offset";
	break;
    case CL_INVALID_EVENT_WAIT_LIST:
	errstr = "invalid event wait list";
	break;
    case CL_INVALID_EVENT:
	errstr = "invalid event";
	break;
    case CL_INVALID_OPERATION:
	errstr = "invalid operation";
	break;
    case CL_INVALID_GL_OBJECT:
	errstr = "invalid GL object";
	break;
    case CL_INVALID_BUFFER_SIZE:
	errstr = "invalid buffer size";
	break;
    case CL_INVALID_MIP_LEVEL:
	errstr = "invalid MIP level";
	break;
    case CL_INVALID_GLOBAL_WORK_SIZE:
	errstr = "invalid global work size";
	break;
    }
    fprintf(stderr, "OpenCL error in %s: %s\n", msg, errstr);
    exit(1);
}
