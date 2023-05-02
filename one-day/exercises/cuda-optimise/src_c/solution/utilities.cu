/*
 * This is a CUDA code that performs an iterative reverse edge 
 * detection algorithm.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2013 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/types.h>
#include <sys/time.h>


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

  if (NULL == (fp = fopen(filename,"r")))
  {
    fprintf(stderr, "datread: cannot open <%s>\n", filename);
    exit(-1);
  }

  fscanf(fp,"%d %d",&nxt,&nyt);

  if (nx != nxt || ny != nyt)
  {
    fprintf(stderr,
            "datread: size mismatch, (nx,ny) = (%d,%d) expected (%d,%d)\n",
            nxt, nyt, nx, ny);
    exit(-1);
  }

  for (j=0; j<ny; j++)
  {
    for (i=0; i<nx; i++)
    {
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

  if (NULL == (fp = fopen(filename,"w")))
  {
    fprintf(stderr, "pgmwrite: cannot create <%s>\n", filename);
    exit(-1);
  }

  /*
   *  Find the max and min absolute values of the array
   */

  xmin = fabs(x[0]);
  xmax = fabs(x[0]);

  for (i=0; i < nx*ny; i++)
  {
    if (fabs(x[i]) < xmin) xmin = fabs(x[i]);
    if (fabs(x[i]) > xmax) xmax = fabs(x[i]);
  }

  fprintf(fp, "P2\n");
  fprintf(fp, "# Written by pgmwrite\n");
  fprintf(fp, "%d %d\n", nx, ny);
  fprintf(fp, "%d\n", (int) thresh);

  k = 0;

  for (j=ny-1; j >=0 ; j--)
  {
    for (i=0; i < nx; i++)
    {
      /*
       *  Access the value of x[i][j]
       */

      tmp = x[j*nx+i];

      /*
       *  Scale the value appropriately so it lies between 0 and thresh
       */

      if (xmin < 0 || xmax > thresh)
      {
        tmp = (int) ((thresh*((fabs(tmp-xmin))/(xmax-xmin))) + 0.5);
      }
      else
      {
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

/* Simple utility function to check for CUDA runtime errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

