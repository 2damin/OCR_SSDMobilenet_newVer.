
#ifndef __CUDACC_RTC__
#define __CUDACC_RTC__
#endif
#include <device_functions.h>
#include <math.h>
#include <cuda.h>
#include "anchorBox_generator.cuh"
#include "error_util.h"

const int cacheMemory = 1000;

__global__ void generate(float* aspect_ratio_layer0, float* aspect_ratio, float* scales_layer0, const float minScale, const float maxScale, const int anchorChannel, float* anchorShapeLayer0, float* anchorShapeLayer1to5) 
{
	__shared__ float scales[cacheMemory];
	__shared__ float aspect_ratio_gpu[cacheMemory];

	int idxRow = threadIdx.x + blockDim.x*blockIdx.x;
	int idxCol = threadIdx.y + blockDim.y*blockIdx.y;

	if (idxRow < anchorChannel) {
		scales[idxRow] = minScale + (maxScale - minScale)*idxRow / (anchorChannel - 1);
		scales[6] = 1;
		aspect_ratio_gpu[idxRow] = aspect_ratio[idxRow];
	}

	__syncthreads();

	int thread_even = min(threadIdx.x * 2, 58);
	int thread_odd = min(threadIdx.x * 2 + 1, 59);

	anchorShapeLayer1to5[thread_even] = scales[thread_even / (anchorChannel * 2) +1] / sqrt((double)aspect_ratio_gpu[(thread_even % (anchorChannel *2)) / 2]);
	anchorShapeLayer1to5[thread_odd] = scales[thread_odd / (anchorChannel * 2) +1] * sqrt((double)aspect_ratio_gpu[(thread_odd % (anchorChannel * 2)) / 2]);
	
	if (thread_even % (anchorChannel * 2) == 10)
	{
		anchorShapeLayer1to5[thread_even] = sqrt(scales[thread_even / (anchorChannel * 2) + 1] * scales[thread_even / (anchorChannel * 2) + 2]) / sqrt((double)aspect_ratio_gpu[(thread_even % (anchorChannel * 2)) / 2]);
		anchorShapeLayer1to5[thread_odd] = sqrt(scales[thread_odd / (anchorChannel * 2) + 1] * scales[thread_odd / (anchorChannel * 2) + 2]) * sqrt((double)aspect_ratio_gpu[(thread_odd % (anchorChannel * 2)) / 2]);
	}

	int thread_even2 = min(thread_even, 4);
	int thread_odd2 = min(thread_odd, 5);

	anchorShapeLayer0[thread_even2] = scales_layer0[thread_even2 / 2] / sqrt((double)aspect_ratio_layer0[(thread_even2 % anchorChannel) / 2]);
	anchorShapeLayer0[thread_odd2] = scales_layer0[thread_odd2 / 2] * sqrt((double)aspect_ratio_layer0[(thread_odd2 % anchorChannel) / 2]);
}


void anchorbox_generate(float* aspect_ratio_layer0 ,float* aspect_ratio, float* scales_layer0, const float minScale, const float maxScale, const int anchorChannel, float* anchorShapeLayer0, float* anchorShapeLayer1to5, dim3 threads_per_block, dim3 num_of_blocks) {
	generate << <num_of_blocks, threads_per_block >> > (aspect_ratio_layer0, aspect_ratio, scales_layer0, minScale, maxScale, anchorChannel, anchorShapeLayer0, anchorShapeLayer1to5);
	cudaDeviceSynchronize();
	return;
}