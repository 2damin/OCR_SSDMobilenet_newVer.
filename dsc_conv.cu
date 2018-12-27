#include <math.h>
#include <cuda.h>
#include "dsc_conv.cuh"
#include "error_util.h"

__global__ void depthwise_conv(float *image, float *tmp, float *depthwiseResult, float *filter, const int input_h, const int input_w, const int depth, const int depthFilterW,
	const int depthwiseOutputW, const int padding, const int stride, const int blocksPerChannel)
{
	int idxRow = threadIdx.x + blockDim.x * blockIdx.x;
	int idxCol = threadIdx.y + blockDim.y * blockIdx.y;
	int idxXperChannel = idxRow % (blockDim.x * blocksPerChannel);
	int idxYperChannel = idxCol % (blockDim.y * blocksPerChannel);
	int idxChannel = ((int)(blockIdx.x) / blocksPerChannel) % depth;


	if (idxXperChannel <depthwiseOutputW && idxYperChannel < depthwiseOutputW)
	{
		for (int k = 0; k < depthFilterW; ++k) {
			for (int q = 0; q < depthFilterW; ++q) {
				depthwiseResult[idxChannel *depthwiseOutputW*depthwiseOutputW + depthwiseOutputW * idxYperChannel + idxXperChannel] 
					+= tmp[idxChannel *input_h*input_w + input_w*(k + idxYperChannel* stride) + (q + idxXperChannel*stride)]
					* filter[idxChannel * depthFilterW*depthFilterW + k*depthFilterW + q];
			}
		}
	}

	//int depthBoundary = idxCol / depthwiseOutputW;
	//int height_per_channel = idxCol % depthwiseOutputW;
	//int tmpSize = col + padding;
	//if (idxRow <depthwiseOutputW && idxCol < depthwiseOutputW * depth)
	//{	
	//	for (int k = 0; k < depthFilterW; ++k) {
	//		for (int q = 0; q < depthFilterW; ++q) {
	//			//depthwiseResult[depthwiseOutputW * idxCol + idxRow] += image[depthBoundary * (int)powf(col, 2) + col*(k + idxCol*stride - depthBoundary * (depthwiseOutputW*stride))
	//			//	+ (q + idxRow*stride)] * filter[depthBoundary * (int)powf(depthFilterW, 2) + k*depthFilterW + q];
	//			depthwiseResult[depthwiseOutputW * idxCol + idxRow] += tmp[depthBoundary *tmpSize*tmpSize + tmpSize*(k + height_per_channel* stride)+ (q + idxRow*stride)]
	//				* filter[depthBoundary * depthFilterW*depthFilterW + k*depthFilterW + q];
	//		}
	//	}
	//	//printf("idxRow: %d, idxCol: %d \n %d \n ------ \n", idxRow, idxCol, (int)powf(col,2));
	//	
	//	//printf("w : %d \n", idxRow*stride);
	//}
}

void depth_f(float *image, float *tmp, float *depthwiseResult, float *filter, const int input_h, const int input_w, const int depth, const int depthFilterW, const int depthwiseOutputW,
	const int padding, const int stride, const int conv_blocks_num, dim3 threads_per_block, dim3 num_of_blocks) {

	depthwise_conv << < num_of_blocks, threads_per_block  >> > (image, tmp, depthwiseResult, filter,
		input_h, input_w, depth, depthFilterW, depthwiseOutputW, padding, stride, conv_blocks_num);
	cudaDeviceSynchronize();
	return;
}

__global__ void pointwise_conv(float *depthwiseResult,  float *pointwiseResult, float *pointwiseFilter, const int col, const int depth, const int depthFilterW,
	const int depthwiseOutputW, const int outputDepth, const int padding, const int stride, const int blocksPerChannel)
{
	//const int depthwiseOutputW = (col - depthFilterW + 2 * padding) / stride + 1;
	int idxRow = threadIdx.x + blockDim.x * blockIdx.x;
	int idxCol = threadIdx.y + blockDim.y * blockIdx.y;
	//int depthBoundary = idxCol / depthwiseOutputW;

	int idxXperChannel = idxRow % (blockDim.x * blocksPerChannel);
	int idxYperChannel = idxCol % (blockDim.y * blocksPerChannel);
	int idxN = ((int)(blockIdx.x) / blocksPerChannel);

	if (idxXperChannel < depthwiseOutputW && idxYperChannel < depthwiseOutputW)
	{
		for (int k = 0; k < depth; ++k)
		{
			//pointwiseResult[idxCol * depthwiseOutputW + idxRow] = 0.0;
			pointwiseResult[idxN *depthwiseOutputW*depthwiseOutputW + idxYperChannel * depthwiseOutputW + idxXperChannel] 
				+= depthwiseResult[k *depthwiseOutputW*depthwiseOutputW + idxYperChannel * depthwiseOutputW + idxXperChannel]
				* pointwiseFilter[k + idxN * depth];
			//std::cout << "pointwiseFilter : " << pointwiseFilter[k + depthBoundary * depth] << std::endl;
		}
	}
}

__global__ void pointwise_conv_bias(float *depthwiseResult, float *pointwiseResult, float *pointwiseFilter, float *bias, const int col, const int depth, const int depthFilterW,
	const int depthwiseOutputW, const int outputDepth, const int padding, const int stride, const int blocksPerChannel)
{
	int idxRow = threadIdx.x + blockDim.x * blockIdx.x;
	int idxCol = threadIdx.y + blockDim.y * blockIdx.y;
	int idxXperChannel = idxRow % (blockDim.x * blocksPerChannel);
	int idxYperChannel = idxCol % (blockDim.y * blocksPerChannel);
	int idxN = ((int)(blockIdx.x) / blocksPerChannel);

	if (idxXperChannel < depthwiseOutputW && idxYperChannel < depthwiseOutputW)
	{
		for (int k = 0; k < depth; ++k)
		{
			pointwiseResult[idxN * depthwiseOutputW*depthwiseOutputW + idxYperChannel * depthwiseOutputW + idxXperChannel]
				+= depthwiseResult[k *depthwiseOutputW*depthwiseOutputW + idxYperChannel * depthwiseOutputW + idxXperChannel]
				* pointwiseFilter[k + idxN * depth];
		}
		pointwiseResult[idxN * depthwiseOutputW*depthwiseOutputW + idxYperChannel * depthwiseOutputW + idxXperChannel] += bias[idxN];
	}
	else {}
}

void point_f(float *depthwiseResult, float *pointwiseResult, float *pointwiseFilter, const int col, const int depth, const int depthFilterW, const int depthwiseOutputW,
	const int outputDepth, const int padding, const int stride, const int conv_blocks_num, dim3 threads_per_block, dim3 num_of_blocks) {

	pointwise_conv << < num_of_blocks, threads_per_block >> > (depthwiseResult, pointwiseResult, pointwiseFilter, col, depth, depthFilterW, depthwiseOutputW, outputDepth, padding, stride, conv_blocks_num);
	cudaDeviceSynchronize();
	return;
}

void pointbias_f(float *depthwiseResult, float *pointwiseResult, float *pointwiseFilter, float *bias, const int col, const int depth, const int depthFilterW, const int depthwiseOutputW,
	const int outputDepth, const int padding, const int stride, const int conv_blocks_num, dim3 threads_per_block, dim3 num_of_blocks) {

	pointwise_conv_bias << < num_of_blocks, threads_per_block >> > (depthwiseResult, pointwiseResult, pointwiseFilter, bias, col, depth, depthFilterW, depthwiseOutputW, outputDepth, padding, stride, conv_blocks_num);
	cudaDeviceSynchronize();
	return;
}




