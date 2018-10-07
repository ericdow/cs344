/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include "utils.h"

__global__ 
void compute_coarse_bins(const unsigned int* const d_vals,
    unsigned short* const d_coarse_bin_ids, int numElems,
    int numCoarseBins) 
{
  // Use a coarse bin ID based on each values percentile
  // Mean is 500, Std. dev. is 100
  volatile int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < numElems) {
    float cdf = 0.5f * (1.0f + erf((d_vals[tid] - 500.0f) / 
        141.421f));
    d_coarse_bin_ids[tid] = (unsigned short) (cdf * (float) numCoarseBins);
  }
}

__global__ void compute_num_zeros(unsigned short* const d_in_coarse_bin_ids,
    const size_t numElems, const int bit, 
    unsigned int* const d_num_zeros) {

  volatile int tid = threadIdx.x + blockDim.x * blockIdx.x;
  
  __shared__ int num_zeros;
  if (tid < numElems) {
    num_zeros = 0;
    __syncthreads();
    
    // Compute local number of zeros for this block
    if (!(d_in_coarse_bin_ids[tid] & (1 << bit))) // bit is 0
      atomicAdd(&num_zeros, 1);
    __syncthreads(); 
      
    // Add the local value to the global value
    if (threadIdx.x == 0) 
      atomicAdd(d_num_zeros, num_zeros);
  } 
}

__global__ void compute_predicate(
    const unsigned short* const d_in_coarse_bin_ids,
    unsigned int* const d_pred0, unsigned int* d_pred1, const size_t numElems, 
    const int bit) {
  volatile int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < numElems) {
    if (d_in_coarse_bin_ids[tid] & (1 << bit)) { // bit is 1
      d_pred0[tid] = 0;
      d_pred1[tid] = 1;
    }
    else { // bit is 0
      d_pred0[tid] = 1;
      d_pred1[tid] = 0;
    }
  }  
}

__global__ void compute_offset(const unsigned int* const d_pred, 
    unsigned int* const d_offset, const size_t start, const size_t numElems) {
  extern __shared__ unsigned int tmp[];

  volatile int tid = threadIdx.x;

  if (start + tid >= numElems)
    return;

  if (tid == 0) {
    if (start == 0)
      tmp[tid] = 0;
    else
      tmp[tid] = d_offset[start-1] + d_pred[start-1];
  }
  else {
    tmp[tid] = d_pred[start + (tid-1)];
  }
  __syncthreads();

  for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
    if (tid >= offset) {
      unsigned int left = tmp[tid - offset];
      __syncthreads();
      tmp[tid] += left;
    }
    __syncthreads();
  }

  d_offset[start + tid] = tmp[tid];
}

__global__ void sort_bit(const unsigned short* const d_in_coarse_bin_ids, 
    unsigned int* const d_in_vals, 
    unsigned short* const d_out_coarse_bin_ids, 
    unsigned int* const d_out_vals, 
    const size_t numElems, const int bit, const unsigned int* const d_num_zeros,
    const unsigned int* const d_offset0, const unsigned int* const d_offset1) {
  
  volatile int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < numElems) {
    volatile int new_pos;
    if (d_in_coarse_bin_ids[tid] & (1 << bit)) { // bit is 1
      new_pos = *d_num_zeros + d_offset1[tid];
    }
    else { // bit is 0
      new_pos = d_offset0[tid];
    }
    d_out_coarse_bin_ids[new_pos] = d_in_coarse_bin_ids[tid];
    d_out_vals[new_pos] = d_in_vals[tid];
  }
}

__global__
void yourHisto(const unsigned int* const d_vals, //INPUT
               unsigned int* const d_histo,      //OUPUT
               int numElems)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //////////////////////////////////////////////////////////////////
  // Compute coarse bin ID for each value
  //////////////////////////////////////////////////////////////////
  const dim3 blockSize(512, 1, 1);
  const dim3 gridSize(floor(numElems/blockSize.x)+1, 1, 1);

  unsigned short * d_coarse_bin_ids;
  checkCudaErrors(cudaMalloc((void **) &d_coarse_bin_ids, 
        sizeof(unsigned short) * numElems));

  // Choose number of coarse bins to fit into shared memory
  int memSize = 49152 / sizeof(unsigned int) / 2;
  int numCoarseBins = numElems / memSize;
  compute_coarse_bins<<<gridSize, blockSize>>>(d_vals, d_coarse_bin_ids,
      numElems, numCoarseBins);

  /*
  unsigned short* h_bin_ids = new unsigned short[numElems];   
  unsigned int* h_vals = new unsigned int[numElems];   
  checkCudaErrors(cudaMemcpy(h_bin_ids, d_coarse_bin_ids, 
        sizeof(unsigned short) * numElems, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_vals, d_vals, 
        sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < numElems; ++i) {
    std::cout << h_vals[i] << " " << h_bin_ids[i] << std::endl;
  }
  delete[] h_bin_ids;
  delete[] h_vals;
  */

  //////////////////////////////////////////////////////////////////
  // Sort the values by the coarse bin ID
  //////////////////////////////////////////////////////////////////
  unsigned int * d_pred0, * d_pred1, * d_offset0, * d_offset1;
  checkCudaErrors(cudaMalloc((void **) &d_pred0, 
        numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_pred1, 
        numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_offset0, 
        numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_offset1, 
        numElems * sizeof(unsigned int)));

  unsigned short * d_coarse_bin_ids_tmp;
  checkCudaErrors(cudaMalloc((void **) &d_coarse_bin_ids_tmp, 
        sizeof(unsigned short) * numElems));

  unsigned int * d_vals_1, * d_vals_2;
  checkCudaErrors(cudaMalloc((void **) &d_vals_1, 
        sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc((void **) &d_vals_2, 
        sizeof(unsigned int) * numElems));
  cudaMemcpy(d_vals_1, d_vals, sizeof(unsigned int) * numElems,
      cudaMemcpyDeviceToDevice);
    
  unsigned int * d_num_zeros;
  checkCudaErrors(cudaMalloc((void **) &d_num_zeros, 
        sizeof(unsigned int)));

  // Determine how many bits actually need to be sorted
  // TODO why doesn't this work???
  int num_bits_to_sort = floor(log2((double)numCoarseBins)) + 1;
  num_bits_to_sort = 16;

  for (int b = 0; b < num_bits_to_sort; ++b) {
    unsigned int* d_in_vals, * d_out_vals;
    unsigned short* d_in_coarse_bin_ids, * d_out_coarse_bin_ids;
    if (b % 2 == 0) {
      d_in_coarse_bin_ids = d_coarse_bin_ids;
      d_in_vals = d_vals_1;
      d_out_coarse_bin_ids = d_coarse_bin_ids_tmp;
      d_out_vals = d_vals_2;
    }
    else {
      d_in_coarse_bin_ids = d_coarse_bin_ids_tmp;
      d_in_vals = d_vals_2;
      d_out_coarse_bin_ids = d_coarse_bin_ids;
      d_out_vals = d_vals_1;
    }

    // Compute the number of zeros for this bit
    checkCudaErrors(cudaMemset(d_num_zeros, 0, sizeof(unsigned int)));
    compute_num_zeros<<<gridSize, blockSize>>>(d_in_coarse_bin_ids, numElems, 
        b, d_num_zeros);

    // Compute the predicate results for this bit
    compute_predicate<<<gridSize, blockSize>>>(d_in_coarse_bin_ids, d_pred0, 
        d_pred1, numElems, b);

    // Determine the relative offset for 0-bit values
    int num_sub_scans = floor(numElems/blockSize.x)+1;
    for (int s = 0; s < num_sub_scans; ++s) {
      compute_offset<<<1, blockSize, blockSize.x * sizeof(unsigned int)>>>(
          d_pred0, d_offset0, s*blockSize.x, numElems);
    }
    
    // Determine the relative offset for 1-bit values
    for (int s = 0; s < num_sub_scans; ++s) {
      compute_offset<<<1, blockSize, blockSize.x * sizeof(unsigned int)>>>(
          d_pred1, d_offset1, s*blockSize.x, numElems);
    }

    // Sort this bit
    sort_bit<<<gridSize, blockSize>>>(d_in_coarse_bin_ids, d_in_vals, 
        d_out_coarse_bin_ids, d_out_vals,
        numElems, b, d_num_zeros, d_offset0, d_offset1);
  }

  cudaFree(d_pred0);
  cudaFree(d_pred1);
  cudaFree(d_offset0);
  cudaFree(d_offset1);
  cudaFree(d_coarse_bin_ids_tmp);
  cudaFree(d_vals_2);
  cudaFree(d_num_zeros);

  // Note: sorted bins/vals stored in d_coarse_bin_ids, d_vals_1 

  /*
  unsigned short* h_bin_ids = new unsigned short[numElems];   
  unsigned int* h_vals = new unsigned int[numElems];   
  checkCudaErrors(cudaMemcpy(h_bin_ids, d_coarse_bin_ids, 
        sizeof(unsigned short) * numElems, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_vals, d_vals_1, 
        sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < numElems; ++i) {
    std::cout << h_vals[i] << " " << h_bin_ids[i] << std::endl;
  }
  delete[] h_bin_ids;
  delete[] h_vals;
  */

  //////////////////////////////////////////////////////////////////
  // Find where each coarse bin begins/ends
  //////////////////////////////////////////////////////////////////
  // TODO

  cudaFree(d_coarse_bin_ids);
  cudaFree(d_vals_1);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
