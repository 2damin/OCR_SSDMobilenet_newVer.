
#ifndef __CUDACC_RTC__
#define __CUDACC_RTC__
#endif
#include <device_functions.h>
#include <math.h>
#include <cuda.h>
#include "error_util.h"
#include "Standard_convolution.cuh"

using namespace std;

void resize(int size, float **data)
{
	if (*data != NULL)
	{
		checkCudaErrors(cudaFree(*data));
	}
	checkCudaErrors(cudaMalloc(data, size * sizeof(float)));
	checkCudaErrors(cudaMemset(*data, 0, size * sizeof(float)));
	return;
}

__global__ void conv(float *srcData, float* tmpConvData, float* dstData, float* filter, const int col, const int row, const int channel, const int n, const int filtersize, const int convOutputW, const int padding, const int stride, int blocksPerChannel) {
	int idxRow = threadIdx.x + blockDim.x * blockIdx.x;
	int idxCol = threadIdx.y + blockDim.y * blockIdx.y;
	int idxXperChannel = idxRow % (blockDim.x * blocksPerChannel);
	int idxYperChannel = idxCol % (blockDim.y * blocksPerChannel);
	int idxXSrcPerChannel = (int)fminf((float)(idxXperChannel), (float)convOutputW);
	int idxYSrcPerChannel = (int)fminf((float)(idxYperChannel), (float)convOutputW);
	int idxChannel = ((int)(blockIdx.x) / blocksPerChannel) % channel;
	int idxN = ((int)(blockIdx.x) / blocksPerChannel) / channel;
	
	if (idxXperChannel < convOutputW && idxYperChannel < convOutputW)
	{
		
		for (int j = 0; j < filtersize; j++) {
			for (int i = 0; i < filtersize; i++) {
				tmpConvData[idxN * convOutputW * convOutputW * channel + idxChannel * convOutputW * convOutputW + idxYperChannel * convOutputW + idxXperChannel]
					+= srcData[idxChannel * col * row + (idxYperChannel * stride + j) * row + (idxXperChannel * stride + i)]
					* filter[idxN * filtersize * filtersize * channel + idxChannel * filtersize * filtersize + j * filtersize + i];
			}
		}

		//dstData[idxN * convOutputW * convOutputW + idxYSrcPerChannel * convOutputW + idxXSrcPerChannel] 
		//	+= tmpConvData[idxN*convOutputW*convOutputW*channel + idxChannel * convOutputW * convOutputW + +idxYSrcPerChannel * convOutputW + idxXSrcPerChannel];
	}

}

__global__ void sum_conv(float* tmpConvData, float* dstData, float* filter, const int col, const int row, const int channel, const int n, const int filtersize, const int convOutputW, const int padding, const int stride, int blocksPerChannel) {
	int idxRow = threadIdx.x + blockDim.x * blockIdx.x;
	int idxCol = threadIdx.y + blockDim.y * blockIdx.y;
	int idxXperChannel = idxRow % (blockDim.x * blocksPerChannel);
	int idxYperChannel = idxCol % (blockDim.y * blocksPerChannel);
	int idxXSrcPerChannel = (int)fminf((float)(idxXperChannel), (float)convOutputW);
	int idxYSrcPerChannel = (int)fminf((float)(idxYperChannel), (float)convOutputW);
	//int idxChannel = ((int)(blockIdx.x) / blocksPerChannel) % channel;
	int idxN = ((int)(blockIdx.x) / blocksPerChannel);

	if (idxXperChannel < convOutputW && idxYperChannel < convOutputW)
	{
		for (int i = 0; i < channel; i++) {
			dstData[idxN * convOutputW * convOutputW + idxYperChannel * convOutputW + idxXperChannel]
				+= tmpConvData[idxN*convOutputW*convOutputW*channel + i * convOutputW * convOutputW + idxYperChannel * convOutputW + idxXperChannel];
		}
	}

}

__global__ void paddingset_conv(float *image, float *tmp, const int depthwiseOutputW, const int col, const int depth, const int padding_topLeft, const int padding_bottomRight)
{
	int idxRow = threadIdx.x + blockDim.x * blockIdx.x;
	int idxCol = threadIdx.y + blockDim.y * blockIdx.y;
	int depthBoundary = idxCol / col;
	int tmpSize = col + padding_topLeft + padding_bottomRight;
	//printf("(%d, %d) \n", threadIdx.y, blockIdx.y);
	if (idxRow < col && idxCol < col * depth)
	{
		tmp[depthBoundary*tmpSize*tmpSize + (idxCol - depthBoundary*col + padding_topLeft)*(tmpSize)+(idxRow + padding_topLeft)] = image[depthBoundary*col*col + (idxCol - depthBoundary*col)* col + idxRow];

	}
}

__global__ void batchNorm(float* srcData, float* dstData, float* batchBias, float* batchScale, float* batchMean, float* batchVar, float epsilon, int channel, int h, int w, int blocksPerChannel) {
	int idxRow = threadIdx.x + blockDim.x * blockIdx.x;
	int idxCol = threadIdx.y + blockDim.y * blockIdx.y;
	int idxXperChannel = idxRow % (blockDim.x * blocksPerChannel);
	int idxYperChannel = idxCol % (blockDim.y * blocksPerChannel);
	int idxXSrcPerChannel = (int)fminf((float)(idxXperChannel), (float)w);
	int idxYSrcPerChannel = (int)fminf((float)(idxYperChannel), (float)h);
	int idxChannel = ((int)(blockIdx.x) / blocksPerChannel);
	if (idxXperChannel < w && idxYperChannel < h)
	{
		dstData[idxChannel * h * w + idxYperChannel * w + idxXperChannel]
			= batchScale[idxChannel] * (srcData[idxChannel* h * w + idxYperChannel * w + idxXperChannel] - batchMean[idxChannel])* (float)rsqrtf((double)epsilon + (double)batchVar[idxChannel]) + batchBias[idxChannel];
		
	}
}

void padding_conv(float *image, float *tmp, const int depthwiseOutputW, const int col, const int depth, const int padding_topLeft, const int padding_bottomRight, dim3 threads_per_block, dim3 num_of_blocks) {
	paddingset_conv << <  num_of_blocks, threads_per_block >> > (image, tmp, depthwiseOutputW, col, depth, padding_topLeft, padding_bottomRight);
	return;
}

void convolution(float **srcData, float **tmpData, float **dstData,float ** filter, const int col, const int row, const int channel, const int n, const int filterSize, const int padding, const int stride) {
	const int convOutputW = (int)(col - filterSize + 2 * padding) / stride + 1;
	const int padding_along_hw = (int)fmaxf((float)((convOutputW - 1)*stride + filterSize - col), 0);
	const int padding_topLeft = (int)(padding_along_hw / 2);
	const int padding_bottomRight = padding_along_hw - padding_topLeft;
	const int padding_blocks = col*channel / 32 + 1;
	const int paddedW = padding_along_hw + row;
	const int paddedH = padding_along_hw + col;
	dim3 theadsPerBlock_padding(32, 32, 1);
	dim3 numOfBlocks_padding(padding_blocks, padding_blocks, 1);
	float* tmpConvData = NULL;
	resize(channel * paddedH*paddedW, tmpData);

	padding_conv(*srcData, *tmpData, convOutputW, col, channel, padding_topLeft, padding_bottomRight, theadsPerBlock_padding, numOfBlocks_padding);

	resize(n * channel * convOutputW *convOutputW, &tmpConvData);

	int blocksPerChannel = (int)convOutputW / 30 + 1;

	dim3 threadsPerBlock(30, 30, 1);
	dim3 numOfBlocks(blocksPerChannel * channel * n, blocksPerChannel, 1);
	conv << <numOfBlocks, threadsPerBlock >> > (*tmpData, tmpConvData, *dstData, *filter, paddedH, paddedW, channel, n, filterSize, convOutputW, padding, stride, blocksPerChannel);
	cudaDeviceSynchronize();

	dim3 numOfBlocks_sumConv(blocksPerChannel * n, blocksPerChannel, 1);
	sum_conv << <numOfBlocks_sumConv, threadsPerBlock >> > (tmpConvData, *dstData, *filter, paddedH, paddedW, channel, n, filterSize, convOutputW, padding, stride, blocksPerChannel);
	cudaDeviceSynchronize();

	//float* tmpSum = new float[n*channel*convOutputW*convOutputW];
	//memset(tmpSum, 0, sizeof(float)*n*channel*convOutputW*convOutputW);
	//float* tmpConv = new float[n*convOutputW*convOutputW];
	//memset(tmpConv, 0, sizeof(float)*n*convOutputW*convOutputW);
	//checkCudaErrors(cudaMemcpy(tmpSum, tmpConvData, sizeof(float)*n * channel * convOutputW * convOutputW, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(tmpConv, *dstData, sizeof(float)*n * convOutputW * convOutputW, cudaMemcpyDeviceToHost));
	//float tmpsum2 = 0;
	//for (int q = 0; q < n; q++) {
	//	for (int i = 0; i < convOutputW; i++) {
	//		for (int j = 0; j < convOutputW; j++) {
	//			for (int k = 0; k < channel; k++) {
	//				tmpsum2 += tmpSum[q*channel*convOutputW*convOutputW + k * convOutputW*convOutputW + i * convOutputW + j];
	//				//cout << tmpSum[q*channel*convOutputW*convOutputW + k * convOutputW*convOutputW + i * convOutputW + j] << "\t";
	//			}if (fabsf(tmpsum2 - tmpConv[q* convOutputW*convOutputW + i * convOutputW + j]) > 0.001) {
	//				cout << "ERROR" << endl; system("pause"); 
	//			}
	//			tmpsum2 = 0;
	//		}
	//	}cout << "==================================" << endl;
	//}


	//cout << "====ConvOut====" << endl;
	//float* tmpDsc_conv = new float[convOutputW * convOutputW * n];
	//memset(tmpDsc_conv, 0, sizeof(float) *convOutputW * convOutputW * n);
	//checkCudaErrors(cudaMemcpy(tmpDsc_conv, *dstData, sizeof(float)* convOutputW * convOutputW * n, cudaMemcpyDeviceToHost));//srcData
	//for (int l = 0; l < 1; ++l) {
	//	for (int k = 0; k < 1; ++k) {
	//		for (int j = 1; j < 2; ++j) {
	//			for (int i = 1; i < 2; ++i) {
	//				cout << tmpDsc_conv[k * convOutputW*convOutputW + convOutputW * j + i] << "\t";
	//			}
	//			//printf("\n");
	//		}
	//		printf("----------------------\n");
	//	}
	//	printf("=================\n");
	//}

	checkCudaErrors(cudaFree(tmpConvData));
	return;
}

void batchNormalization(float* srcData, float* dstData, float* batchBias, float* batchScale, float* batchMean, float* batchVar, float epsilon, int channel, int h, int w) {
	int blocksPerChannel = (int)h / 30 + 1;
	dim3 threadsPerBlock(30, 30, 1);
	dim3 numOfBlocks(blocksPerChannel * channel, blocksPerChannel, 1);

	batchNorm<<<numOfBlocks, threadsPerBlock>>>(srcData, dstData, batchBias, batchScale, batchMean, batchVar, epsilon, channel, h, w, blocksPerChannel);
	cudaDeviceSynchronize();
	return;
}

