//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <iostream>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void hist(unsigned int* const d_inVals, unsigned int* const d_inPos,
    unsigned int* const d_outVals, unsigned int* const d_outPos, 
    const size_t numElems, const int bit, unsigned int* const d_bins) {
 
  volatile int tid = threadIdx.x + blockDim.x * blockIdx.x;
  
  __shared__ int bins[2];
  if (tid < numElems) {
    bins[0] = 0;
    bins[1] = 0;
    __syncthreads();
    
    if (d_inVals[tid] & (1 << bit)) // bit is 1
      atomicAdd(&(bins[1]), 1);
    else // bit is 0
      atomicAdd(&(bins[0]), 1);
    __syncthreads(); 
    
    // Add the local value to the global value
    if (threadIdx.x == 0) { 
      atomicAdd(&(d_bins[0]), bins[0]);
      atomicAdd(&(d_bins[1]), bins[1]);
    }
  } 
}

__global__ void compute_predicate(const unsigned int* const d_inVals,
    unsigned int* const d_pred0, unsigned int* d_pred1, const size_t numElems, 
    const int bit) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < numElems) {
    if (d_inVals[tid] & (1 << bit)) { // bit is 1
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

  int tid = threadIdx.x;

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

__global__ void sort_bit(unsigned int* const d_inVals, 
    unsigned int* const d_inPos,
    unsigned int* const d_outVals, unsigned int* const d_outPos, 
    const size_t numElems, const int bit, const unsigned int* const d_bins,
    const unsigned int* const d_offset0, const unsigned int* const d_offset1) {
  
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < numElems) {
    if (d_inVals[tid] & (1 << bit)) { // bit is 1
      int new_pos = d_bins[0] + d_offset1[tid];
      d_outVals[new_pos] = d_inVals[tid];
      d_outPos[new_pos] = d_inPos[tid];
    }
    else { // bit is 0
      int new_pos = d_offset0[tid];
      d_outVals[new_pos] = d_inVals[tid];
      d_outPos[new_pos] = d_inPos[tid];
    }
  }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
  const dim3 blockSize(512, 1, 1);
  const dim3 gridSize(floor(numElems/blockSize.x)+1, 1, 1);

  unsigned int * d_bins;
  checkCudaErrors(cudaMalloc((void **) &d_bins, 2 * sizeof(unsigned int)));
    
  unsigned int * d_pred0, * d_pred1, * d_offset0, * d_offset1;
  checkCudaErrors(cudaMalloc((void **) &d_pred0, 
        numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_pred1, 
        numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_offset0, 
        numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_offset1, 
        numElems * sizeof(unsigned int)));
 
  for (int b = 0; b < 32; ++b) {
    // Zero out the histogram
    checkCudaErrors(cudaMemset(d_bins, 0, 2 * sizeof(unsigned int)));

    unsigned int* d_inVals, * d_inPos, * d_outVals, * d_outPos;
    if (b % 2 == 0) {
      d_inVals = d_inputVals;
      d_inPos = d_inputPos;
      d_outVals = d_outputVals;
      d_outPos = d_outputPos;
    }
    else {
      d_inVals = d_outputVals;
      d_inPos = d_outputPos;
      d_outVals = d_inputVals;
      d_outPos = d_inputPos;
    }

    // Compute the histogram for this bit
    hist<<<gridSize, blockSize>>>(d_inVals, d_inPos, d_outVals, d_outPos,
        numElems, b, d_bins);

    // Compute the predicate results for this bit
    compute_predicate<<<gridSize, blockSize>>>(d_inVals, d_pred0, d_pred1,
        numElems, b);

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
    sort_bit<<<gridSize, blockSize>>>(d_inVals, d_inPos, d_outVals, d_outPos,
        numElems, b, d_bins, d_offset0, d_offset1);
  }

  // Copy data back to output
  cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems,
      cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems,
      cudaMemcpyDeviceToDevice);

  cudaFree(d_bins);
  cudaFree(d_pred0);
  cudaFree(d_pred1);
  cudaFree(d_offset0);
  cudaFree(d_offset1);
}
