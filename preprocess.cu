#ifndef __CUDACC_RTC__
#define __CUDACC_RTC__
#endif
#include <device_functions.h>
#include <math.h>
#include <cuda.h>
#include "preprocess.cuh"
#include "error_util.h"

using namespace std;



__global__ void resize(float* srcData, float* dstData, float* xpixelPerBlock, float* ypixelPerBlock, float interpolation_xgap, float interpolation_ygap, int original_image_w, int original_image_h,  int resized_w, int resized_h)
{
	int idxCol = (int)ypixelPerBlock[blockIdx.y] + 1 + threadIdx.y;
	int idxRow = (int)xpixelPerBlock[blockIdx.x] + 1 + threadIdx.x;
	if ((int)xpixelPerBlock[blockIdx.x] == (int)xpixelPerBlock[blockIdx.x + 1] || (int)ypixelPerBlock[blockIdx.y] == (int)ypixelPerBlock[blockIdx.y + 1])
	{}
	else {
	//idxCol = min(idxCol, (int)ypixelPerBlock[blockIdx.y + 1]);
	//idxRow = min(idxRow, (int)xpixelPerBlock[blockIdx.x + 1]);
	idxCol = min(idxCol, (int)((blockIdx.y + 1)*interpolation_ygap));
	idxRow = min(idxRow, (int)((blockIdx.x + 1)*interpolation_xgap));

	float X2 = interpolation_xgap * (blockIdx.x + 1);
	float X1 = interpolation_xgap * blockIdx.x;
	float Y2 = interpolation_ygap * (blockIdx.y + 1);
	float Y1 = interpolation_ygap * blockIdx.y;
	
	float px = ((X2 - (float)idxRow) / (X2 - X1)) * srcData[blockIdx.y * (original_image_w) + blockIdx.x] + (((float)idxRow - X1) / (X2 - X1)) * srcData[blockIdx.y * (original_image_w) + blockIdx.x + 1];
	float qx = ((X2 - (float)idxRow) / (X2 - X1)) * srcData[(blockIdx.y + 1) * (original_image_w) + blockIdx.x] + (((float)idxRow - X1) / (X2 - X1)) * srcData[(blockIdx.y + 1) * (original_image_w) + blockIdx.x + 1];

	dstData[idxCol * resized_w + idxRow] = 2 * (((Y2 - (float)idxCol) / (Y2 - Y1)) * px + (((float)idxCol - Y1) / (Y2 - Y1)) * qx) / float(255) - 1;
	//dstData[idxCol * resized_w + idxRow] = (((Y2 - (float)idxCol) / (Y2 - Y1)) * px + (((float)idxCol - Y1) / (Y2 - Y1)) * qx);
	dstData[resized_w * resized_h * 1 + idxCol * resized_w + idxRow] = 2 * (((Y2 - (float)idxCol) / (Y2 - Y1)) * px + (((float)idxCol - Y1) / (Y2 - Y1)) * qx) / float(255) - 1;
	dstData[resized_w * resized_h * 2 + idxCol * resized_w + idxRow] = 2 * (((Y2 - (float)idxCol) / (Y2 - Y1)) * px + (((float)idxCol - Y1) / (Y2 - Y1)) * qx) / float(255) - 1;

	}
}

__global__ void copyChannel(float* dstData, float* xpixelPerBlock, float* ypixelPerBlock, float interpolation_xgap, float interpolation_ygap,int resized_w, int resized_h)
{
	int idxCol = (int)ypixelPerBlock[blockIdx.y] + 1 + threadIdx.y;
	int idxRow = (int)xpixelPerBlock[blockIdx.x] + 1 + threadIdx.x;
	idxCol = min(idxCol, (int)ypixelPerBlock[blockIdx.y + 1]);
	idxRow = min(idxRow, (int)xpixelPerBlock[blockIdx.x + 1]);

	float X2 = interpolation_xgap * (blockIdx.x + 1);
	float X1 = interpolation_xgap * blockIdx.x;
	float Y2 = interpolation_ygap * (blockIdx.y + 1);
	float Y1 = interpolation_ygap * blockIdx.y;

	dstData[resized_w * resized_h * 1 + idxCol * resized_w + idxRow] = dstData[idxCol * resized_w + idxRow];
	dstData[resized_w * resized_h * 2 + idxCol * resized_w + idxRow] = dstData[idxCol * resized_w + idxRow];

}

void resizeCuda(float* srcData, float* dstData, float* xpixelPerBlock, float* ypixelPerBlock, float interpolation_xgap, float interpolation_ygap, int original_image_w, int original_image_h, int resized_w, int resized_h, dim3 threadsPerBlock, dim3 numOfBlocks)
{
	resize << < numOfBlocks, threadsPerBlock >> > ( srcData, dstData, xpixelPerBlock, ypixelPerBlock, interpolation_xgap, interpolation_ygap, original_image_w, original_image_h, resized_w, resized_h);
	//cudaDeviceSynchronize();

	/*copyChannel <<<numOfBlocks, threadsPerBlock>>>(dstData, xpixelPerBlock, ypixelPerBlock, interpolation_xgap, interpolation_ygap, resized_w, resized_h);
	cudaDeviceSynchronize();*/
	return;
}