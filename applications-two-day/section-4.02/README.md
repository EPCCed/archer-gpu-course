# Graph API

Streams provide a mechanism to control the execution of independent,
or asynchronous, work.

A more general mechanism, added more recently in CUDA, introduces the
idea of a graph.

Graphs can be used to orchestrate complex workflows, and may be
particularly unseful in amortising the overhead of many small
kernel launches.

Note: the latest HIP does support a subset of Graph API operations,
but I haven't had a chance to try it out yet.


## Graphs

The idea (taken from graph theory) is to represent individual executable
items of work by the nodes of the graph, and the dependencies between
them as edges connecting the relevant nodes.

PICTURE

There is the assumption that there is a begining and an end (ie., there
is a direction), and that there are no closed loops in the graph picture.
This gives rise to a *directed acyclic graph*, or DAG.

The idea is then to construct a descipriotn of the graph from the
constituent nodes and dependencies, and then execute the graph.

### Creating a CUDA graph

The overall container for a graph is of type
```
  cudaGraph_t graph;
```
and is allocated in an empty state via the API function
```
  __host__ cudaErr_t cudaGraphCreate(cudaGraph_t * graph, unsigned int flags);
```
The only valid value for the second argument is `flags = 0`.

For example, the life-cycle would typically be:
```
  cudaGraph_t myGraph;

  cudaGraphCreate(&myGraph, 0);

  /* ... work ... */

  cudaGraphDestroy(myGraph);
```
Destroying the graph will also destoy any component nodes/dependencies.


### Instantiating and executing a graph

Having created a graph object, one needs to add nodes and dependencies
to it. When this has been done (adding nodes etc will be discussed below),
one creates an executable graph object
```
  cudaGraphExec_t graphExec;

  cudaGraphInstantiate(&graphExec, myGraph, NULL, NULL, 0);

  /* ... and launch the graph into a stream ... */

  cudaGraphLaunch(graphExec, stream);

  cudaGraphExecDestroy(graphExec);
```
The idea here is that the instantiation step performs a lot of the
overhead of setting up the laucnh parameters and so forth, and then
the luanch is relately small compared with a standard launch.


## Graph definition

The following sections consider the explicit definintion of graph
structure.

### Node types

The nodes of the graph may represent a number of different types of
operation. Valid choices include:

1. A `cudaMemcpy()` operation
2. A `cudaMemset()` operation
3. A kernel
4. A CPU function call

Specifying the nodes of a graph means providing a description of the
arguments which would have been used in a normal invocation, such as
those we have seen for `cudaMemcpy()` before.

### Kernel node

Suppose we have a kernel function with arguments
```
  __global__ void myKernel(double a, double * x);
```
and which is executed with configuration including `blocks` and
`threadsPerBlock`.

These parameters are described in CUDA by a structure `cudaKernelNodeParams`
which included the public members:
```
   void * func;             /* pointer to the kernel function */
   void ** kernelParams;    /* List of kernel dummy arguments */
   dim3 gridDim;            /* Number of blocks */
   dim3 blockDim;           /* Number of threads per block */
```
So, with the relevant host variables in scope, we might write
```
  cudaKernelNodeParams kParams = {0};   /* Initialise to zero */
  void * args[] = {&a, &d_x};           /* Kernel arguments */

  kParams.func         = (void *) myKernel;
  kParams.kernelParams = args;
  kParams.gridDim      = blocks;
  kParams.blockDim     = threadsPerBlock;
```
We are now ready to add a kernel node to the graph (assumed to
be `myGraph`):
```
  cudaGraphNode_t kNode;     /* handle to the new kernel node */

  cudaGraphAddKernelNode(&kNode, myGraph, NULL, 0, &kParams);
```
This creates a new kernel node, adds it to the existing graph, and
returns a handle to the new node.

The formal descrition is
```
__host__ cudaErr_t cudaGraphAddKernelNode(cudaGraphNode_t * node,
                                          cudaGraph_t graph,
					  const cudaGraphNode_t * dependencies,
					  size_t nDependencies,
					  const cudaKernelNodeParams * params);
```
If the new node is not dependent on any other node, then the third and
fourth arguments can be `NULL` and zero, respectively.

### A `memcpy` node

There is a similar procedure to define a `memcpy` node. We need the
structure `cudaMemcpy3DParms` (sic) with relevant public members
```
  struct cudaPos          dstPos;       /* offset in destination */
  struct cudaPitchedPtr   dstptr;       /* address and length in destination */
  struct cudaExtent       extent;       /* dimensions of block */
  cudaMemcpykind          kind;         /* direction of the copy */
  struct cudaPos          srcPos;       /* offset in source */
  struct cudaPitchedPtr   srcPtr;       /* address and length in source */
```
This is rather involved, as it must allow for the most general
type of copy allowed in the CUDA API.

To make this more concrete, consider an explicit `cudaMemcpy()` operation
```
  cudaMempcy(d_ptr, h_ptr, ndata*sizeof(double), cudaMemcpyHostToDevice);
```
We should then define something of the form
```
  cudaGraphNode_t node;
  cudaMemcpy3DParms mParams = {0};

  mParams.kind   = cudaMemcpyHostToDevice;
  mParams.extent = make_cudaExtent(ndata*sizeof(double), 1, 1};
  mParams.srcPos = make_cudaPos(0, 0, 0);
  mParams.srcPtr = make_cudaPitchedPtr(h_ptr, ndata*sizeof(double), ndata, 1);
  mParams.dstPos = make_cudaPos(0, 0, 0);
  mParams.dstPtr = make_cudaPitchedPtr(d_ptr, ndata*sizeof(double), ndata, 1);
```
For simple one-dimensional allocations, it is possible to write some
simple helper functions to hide this complexity.

The information is added via:
```
  cudaGraphAddMemcpyNode(mNode, myGraph, &kNode, 1, &mParams);
```
where we have made it dependent on the preceding kernel node.


These data structures are documented more fully in the data structures
section of the CUDA runtime API.

https://docs.nvidia.com/cuda/cuda-runtime-api/annotated.html#annotated

## Synchronisation

If executing a graph in a particular stream, one can use
```
  cudaStreamSynchronize(stream);
```
to ensure that the graph is complete. It is also possible to use
```
  cudaDeviceSynchronize();
```
which actually synchronises all streams running on the current
device.


## Exercise (30 minutes)

The exercise revisits again the problem for `A_ij := A_ij + x_i y_j`,
and the exercise is to see whether you can replace the single
kernel launch with the execution of a graph. When you have a
working program, check with nsight systems that this is doing what
you expect.

A new template is supplied if you wish to start afresh.

While it will not be possible to see any performance improvement
associated with this single kernel launch, the principle should
be clear.

### Finished?

Have a go at adding to the graph the dependent operation which is the
device to host `cudaMemcpy()` of the result.
