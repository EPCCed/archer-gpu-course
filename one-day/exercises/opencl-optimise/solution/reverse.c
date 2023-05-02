#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <CL/opencl.h>

#include <sys/types.h>
#include <sys/time.h>

/* Forward Declarations of utility functions */
double get_current_time();
void datread(char*, void*, int, int);
void pgmwrite(char*, void*, int, int);
void checkOpenCLError(cl_int, char *);
cl_int initCLDevice(cl_device_type devtype, cl_context *ctxt,
		    cl_command_queue *queue);
cl_int getCLKernel(cl_context ctxt, char *filename, char *funcname,
		   cl_kernel *kernel);

/* Dimensions of image */
#define WIDTH 2048
#define HEIGHT 2048

/* Number of iterations to run */
#define ITERATIONS 10

/* Dimensions of OpenCL local work size */
#define LOCAL_W 16
#define LOCAL_H 16

/* Maximum difference allowed between host result and GPU result */
#define MAX_DIFF 0.01

/* Data buffer to read edge data into */
float edge[HEIGHT][WIDTH];

/* Data buffer for the resulting image */
float img[HEIGHT][WIDTH];

/* Work buffers, with halos */
float host_input[HEIGHT+2][WIDTH+2];
float gpu_output[HEIGHT+2][WIDTH+2];
float host_output[HEIGHT+2][WIDTH+2];


int main(int argc, char *argv[])
{
    int x, y;
    int i;
    int errors;

    double start_time_inc_data, end_time_inc_data;
    double cpu_start_time, cpu_end_time;

    cl_mem d_input, d_output, d_edge, tmp;
    cl_int err;

    int width = WIDTH;
    int height = HEIGHT;

    cl_context ctxt;
    cl_command_queue queue;
    cl_kernel kernel;

    size_t memSize = (WIDTH+2) * (HEIGHT+2) * sizeof(float);

    printf("Image size: %dx%d\n", WIDTH, HEIGHT);
    printf("Local work size: %dx%d\n", LOCAL_W, LOCAL_H);

    /* initialise OpenCL */
    err = initCLDevice(CL_DEVICE_TYPE_GPU, &ctxt, &queue);
    checkOpenCLError(err, "initCLDevice");
    err = getCLKernel(ctxt, "kernels.c", "reverse2d", &kernel);
    //err = getCLKernel(ctxt, "kernels.c", "reverse1d_col", &kernel);
    //err = getCLKernel(ctxt, "kernels.c", "reverse1d_row", &kernel);
    checkOpenCLError(err, "getCLKernel");

    /* allocate memory on device */
    d_input = clCreateBuffer(ctxt, CL_MEM_READ_WRITE, memSize, NULL, &err);
    checkOpenCLError(err, "buffer allocation");
    d_output = clCreateBuffer(ctxt, CL_MEM_READ_WRITE, memSize, NULL, &err);
    checkOpenCLError(err, "buffer allocation");
    d_edge = clCreateBuffer(ctxt, CL_MEM_READ_ONLY, memSize, NULL, &err);
    checkOpenCLError(err, "buffer allocation");
    
    /* read in edge data */
    datread("edge2048x2048.dat", (void *)edge, WIDTH, HEIGHT);
    
    /* zero buffer so that halo is zeroed */
    for (y = 0; y < HEIGHT+2; y++) {
	for (x = 0; x < WIDTH+2; x++) {
	    host_input[y][x] = 0.0;
	}
    }
    
    /* copy input to buffer with halo */
    for (y = 0; y < HEIGHT; y++) {
	for (x = 0; x < WIDTH; x++) {
	    host_input[y+1][x+1] = edge[y][x];
	}
    }
    
    /*
     * copy to all the GPU arrays. d_output doesn't need to have this data but
     * this will zero its halo
     */
    start_time_inc_data = get_current_time();

    err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, memSize, host_input, 0, NULL, NULL);
    checkOpenCLError(err, "buffer write");
    err = clEnqueueWriteBuffer(queue, d_output, CL_TRUE, 0, memSize, host_input, 0, NULL, NULL);
    checkOpenCLError(err, "buffer write");
    err = clEnqueueWriteBuffer(queue, d_edge, CL_TRUE, 0, memSize, host_input, 0, NULL, NULL);
    checkOpenCLError(err, "buffer write");
    
    /* run on GPU */
    for (i = 0; i < ITERATIONS; i++) {

	/* run the kernel */
	/*
	 * One of these kernel invocations should be uncommented at a time. Make sure it
	 * matches the kernel actually loaded above.
	 */

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_output);
	checkOpenCLError(err, "setting kernel arguments");
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_input);
	checkOpenCLError(err, "setting kernel arguments");
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_edge);
	checkOpenCLError(err, "setting kernel arguments");
	err = clSetKernelArg(kernel, 3, sizeof(int), &width);
	checkOpenCLError(err, "setting kernel arguments");

	size_t globalsize[2] = { WIDTH, HEIGHT };
	size_t localsize[2] = { LOCAL_W, LOCAL_H };
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, &globalsize[0], &localsize[0], 0, NULL, NULL);

	//size_t globalsize[1] = { HEIGHT };
	//size_t localsize[1] = { LOCAL_H };
	//err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalsize[0], &localsize[0], 0, NULL, NULL);

	//err = clSetKernelArg(kernel, 4, sizeof(int), &height);
	//checkOpenCLError(err, "setting kernel arguments");
	//size_t globalsize[1] = { WIDTH };
	//size_t localsize[1] = { LOCAL_W };
	//err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalsize[0], &localsize[0], 0, NULL, NULL);

	checkOpenCLError(err, "running kernel");

	/* wait for kernel to complete */
	clFinish(queue);
	
	/* swap the buffer pointers ready for next time */
	tmp = d_input;
	d_input = d_output;
	d_output = tmp;
    }
    
    err = clEnqueueReadBuffer(queue, d_input, CL_TRUE, 0, memSize, gpu_output, 0, NULL, NULL);
    checkOpenCLError(err, "buffer read");
    
    end_time_inc_data = get_current_time();
    
    /*
     * run on host for comparison
     */
    cpu_start_time = get_current_time();
    for (i = 0; i < ITERATIONS; i++) {
	
	/* perform stencil operation */
	for (y = 0; y < HEIGHT; y++) {
	    for (x = 0; x < WIDTH; x++) {
		host_output[y+1][x+1] = (host_input[y+1][x] + host_input[y+1][x+2] +
					 host_input[y][x+1] + host_input[y+2][x+1] \
					 - edge[y][x]) * 0.25;
	    }
	}
	
	/* copy output back to input buffer */
	for (y = 0; y < HEIGHT; y++) {
	    for (x = 0; x < WIDTH; x++) {
		host_input[y+1][x+1] = host_output[y+1][x+1];
	    }
	}
    }
    cpu_end_time = get_current_time();
    
    /* check that GPU result matches host result */
    errors = 0;
    for (y = 0; y < HEIGHT; y++) {
	for (x = 0; x < WIDTH; x++) {
	    float diff = fabs(gpu_output[y+1][x+1] - host_output[y+1][x+1]);
	    if (diff >= MAX_DIFF) {
		errors++;
		printf("Error at %d,%d (CPU=%f, GPU=%f)\n", x, y,	\
		       host_output[y+1][x+1],				\
		       gpu_output[y+1][x+1]);
	    }
	}
    }
    
    if (errors == 0) printf("\n\n ***TEST PASSED SUCCESSFULLY*** \n\n\n");
    
    /* copy result to output buffer */
    for (y = 0; y < HEIGHT; y++) {
	for (x = 0; x < WIDTH; x++) {
	    img[y][x] = gpu_output[y+1][x+1];
	}
    }
    
    /* write PGM */
    pgmwrite("output.pgm", (void *)img, WIDTH, HEIGHT);
    
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_edge);
    
    printf("GPU Time (Including Data Transfer): %fs\n", \
	   end_time_inc_data - start_time_inc_data);
    printf("CPU Time                          : %fs\n", \
	   cpu_end_time - cpu_start_time);
    
    return 0;
}


/* Utility Functions */

/*
 * Function to get an accurate time reading
 */
double get_current_time()
{
    static int start = 0, startu = 0;
    struct timeval tval;
    double result;
    
    if (gettimeofday(&tval, NULL) == -1)
	result = -1.0;
    else if(!start) {
	start = tval.tv_sec;
	startu = tval.tv_usec;
	result = 0.0;
    }
    else
	result = (double) (tval.tv_sec - start) + 1.0e-6*(tval.tv_usec - startu);
    
    return result;
}


/* Read the input file containing the edge data */
void datread(char *filename, void *vx, int nx, int ny)
{ 
    FILE *fp;
    
    int nxt, nyt, i, j, t;
    
    float *x = (float *) vx;
    
    if (NULL == (fp = fopen(filename,"r"))) {
	fprintf(stderr, "datread: cannot open <%s>\n", filename);
	exit(-1);
    }
    
    fscanf(fp,"%d %d",&nxt,&nyt);
    
    if (nx != nxt || ny != nyt) {
	fprintf(stderr,
		"datread: size mismatch, (nx,ny) = (%d,%d) expected (%d,%d)\n",
		nxt, nyt, nx, ny);
	exit(-1);
    }
    
    for (j=0; j<ny; j++) {
	for (i=0; i<nx; i++) {
	    fscanf(fp,"%d", &t);
	    x[(ny-j-1)*nx + i] = t;
	}
    }

    fclose(fp);
}

/* Write the output image as a PGM file */
void pgmwrite(char *filename, void *vx, int nx, int ny)
{
    FILE *fp;
    
    int i, j, k, grey;

    float xmin, xmax, tmp;
    float thresh = 255.0;

    float *x = (float *) vx;

    if (NULL == (fp = fopen(filename,"w"))) {
	fprintf(stderr, "pgmwrite: cannot create <%s>\n", filename);
	exit(-1);
    }

    /*
     *  Find the max and min absolute values of the array
     */
    
    xmin = fabs(x[0]);
    xmax = fabs(x[0]);
    
    for (i=0; i < nx*ny; i++) {
	if (fabs(x[i]) < xmin) xmin = fabs(x[i]);
	if (fabs(x[i]) > xmax) xmax = fabs(x[i]);
    }
    
    fprintf(fp, "P2\n");
    fprintf(fp, "# Written by pgmwrite\n");
    fprintf(fp, "%d %d\n", nx, ny);
    fprintf(fp, "%d\n", (int) thresh);
    
    k = 0;
    
    for (j=ny-1; j >=0 ; j--) {
	for (i=0; i < nx; i++) {
	    /*
	     *  Access the value of x[i][j]
	     */
	    
	    tmp = x[j*nx+i];
	    
	    /*
	     *  Scale the value appropriately so it lies between 0 and thresh
	     */
	    
	    if (xmin < 0 || xmax > thresh) {
		tmp = (int) ((thresh*((fabs(tmp-xmin))/(xmax-xmin))) + 0.5);
	    }
	    else {
		tmp = (int) (fabs(tmp) + 0.5);
	    }
	    
	    /*
	     *  Increase the contrast by boosting the lower values
	     */
	    
	    grey = (int) (thresh * sqrt(tmp/thresh));
	    
	    fprintf(fp, "%3d ", grey);
	    
	    if (0 == (k+1)%16) fprintf(fp, "\n");
	    
	    k++;
	}
    }
    
    if (0 != k%16) fprintf(fp, "\n");
    fclose(fp);
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
