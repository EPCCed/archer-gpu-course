__kernel void reverse1d_col(__global float *d_output, 
                            __global float *d_input,
			    __global float *d_edge, const int width)
{
    int col, row;
    int idx, idx_south, idx_north, idx_west, idx_east;
    int numcols = width + 2;

    /*
     * get global row index for this thread  
     * remember to add 1 to account for halo    
     */
    row = get_global_id(0) + 1;

    /*
     * loop over all columns of the image
     */
    for (col = 1; col <= width; col++) {
        /*
         * calculate linear index from col and row, for the centre
         * and neighbouring points needed below.
         * For the neighbouring points you need to add/subtract 1  
         * to/from the row or col indices.
         */
	
	idx = row * numcols + col;
	idx_south = (row - 1) * numcols + col;
	idx_north = (row + 1) * numcols + col;
	
	idx_west = row * numcols + (col - 1);
	idx_east = row * numcols + (col + 1);
	
        /* perform stencil operation */  
	d_output[idx] = (d_input[idx_south] + d_input[idx_west] + d_input[idx_north] 
			 + d_input[idx_east] - d_edge[idx]) * 0.25;
    }
}

__kernel void reverse1d_row(__global float *d_output, __global float *d_input,
			    __global float *d_edge, const int width, const int height)
{
    int col, row;
    int idx, idx_south, idx_north, idx_west, idx_east;
    int numcols = width + 2;

    /*
     * calculate global column index for this thread  
     * remember to add 1 to account for halo     
     */
    // col = ;

    /*
     * loop over all rows of the image
     */
    // for ( ; ; )
    {
        /*
         * calculate linear index from col and row, for the centre
         * and neighbouring points needed below.
         * For the neighbouring points you need to add/subtract 1  
         * to/from the row or col indices.
         */      
	
	idx = row * numcols + col;
	idx_south = (row - 1) * numcols + col;
	idx_north = (row + 1) * numcols + col;
	
	idx_west = row * numcols + (col - 1);
	idx_east = row * numcols + (col + 1);
	
        /* perform stencil operation */  
	d_output[idx] = (d_input[idx_south] + d_input[idx_west] + d_input[idx_north] 
			 + d_input[idx_east] - d_edge[idx]) * 0.25;
    }
}

__kernel void reverse2d(__global float *d_output, __global float *d_input,
			__global float *d_edge, const int width)
{
    int col, row;
    int idx, idx_south, idx_north, idx_west, idx_east;
    int numcols = width + 2;

    /*
     * get global column index for this thread  
     * remember to add 1 to account for halo     
     */
    // col ;

    /*
     * get global row index for this thread  
     * remember to add 1 to account for halo    
     */
    // row ;

    /*
     * calculate linear index from col and row, for the centre
     * and neighbouring points needed below.
     * For the neighbouring points you need to add/subtract 1  
     * to/from the row or col indices.
     */
    idx = row * numcols + col;
    idx_south = (row - 1) * numcols + col;
    idx_north = (row + 1) * numcols + col;
    
    idx_west = row * numcols + (col - 1);
    idx_east = row * numcols + (col + 1);

    /* perform stencil operation */
    d_output[idx] = (d_input[idx_south] + d_input[idx_west] + d_input[idx_north] 
		     + d_input[idx_east] - d_edge[idx]) * 0.25;
}

