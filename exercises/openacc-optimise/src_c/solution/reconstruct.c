/*
 * This is an OpenACC code that performs an iterative reverse edge 
 * detection algorithm.
 *
 * Training material developed by Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2012 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

/* Forward Declarations of utility functions*/
double get_current_time();
void datread (char *filename, void *vx, int nx, int ny);
void pgmwrite(char *filename, void *vx, int nx, int ny);

/* Dimensions of image */
#define N 2048

/* Number of iterations to run */
#define ITERATIONS   100

float old[N+2][N+2], new[N+2][N+2], edge[N+2][N+2];

float masterbuf[N][N];


int main (int argc, char **argv)
{

  int i, j, iter, maxiter;
  char *filename;

  double start_time, end_time;

  printf("Processing %d x %d image\n", N, N);
  printf("Number of iterations = %d\n", ITERATIONS);

  /* read in edge data */  
  filename = "edge2048x2048.dat";
  printf("\nReading <%s>\n", filename);
  datread(filename, masterbuf, N, N);
  printf("\n");


  /* Initialise variables */
  for (i=0;i<N+2;i++)
    {
      for (j=0;j<N+2;j++)
	{
	  
	  if (i==0 || j==0 || i==N+1 || j==N+1) //if halo
	    // zero halo data
	    edge[i][j]=0.0;
	  else
	    //copy input data
	    edge[i][j]=masterbuf[i-1][j-1];
	  
	  old[i][j]=edge[i][j];
	  
	}
    }
  
  
 
#pragma acc data copy(old) copyin(edge) create(new) 
  { 

  start_time = get_current_time();
           
  /* main loop */
  for (iter=1;iter<=ITERATIONS; iter++)
    {
      
      
#pragma acc parallel vector_length(256) 
    { 
	    
      /* perform stencil operation */
#pragma acc loop 
      for (i=1;i<N+1;i++)
	{
	  for (j=1;j<N+1;j++)
	    {
	      new[i][j]=0.25*(old[i-1][j]+old[i+1][j]+old[i][j-1]+old[i][j+1]
			      - edge[i][j]);
	    }
	}
      
      /* copy output back to input buffer */
#pragma acc loop 
      for (i=1;i<N+1;i++)
	{
	  for (j=1;j<N+1;j++)
	    {
	      old[i][j]=new[i][j];
	    }
	}
      
    }//end parallel region 
    
    
    } /* end of main loop */
      
  end_time = get_current_time();

      
  }//end data region 

    
  printf("Main Loop Time                          : %fs\n",	\
	 end_time - start_time);
  
  
  
  /* copy result to output buffer */
  for (i=1;i<N+1;i++)
    {
      for (j=1;j<N+1;j++)
	{
	  masterbuf[i-1][j-1]=old[i][j];
	}
    }
  
  
    
  printf("\nFinished %d iterations\n", iter-1);
  
  
  filename="image2048x2048.pgm";
  printf("\nWriting <%s>\n", filename);
  pgmwrite(filename, masterbuf, N, N);
  
    
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

