#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void convolution(float **srcData, float **tmpData, float **dstData, float ** filter, const int col, const int row, const int channel, const int n, const int filterSize, const int padding, const int stride);

void padding_conv(float *image, float *tmp, const int depthwiseOutputW, const int col, const int depth, const int padding_topLeft, const int padding_bottomRight, dim3 threads_per_block, dim3 num_of_blocks);

void batchNormalization(float* srcData, float* dstData, float* batchBias, float* batchScale, float* batchMean, float* batchVar, float epsilon, int channel, int h, int w);