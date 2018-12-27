#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void resizeCuda(float* srcData, float* dstData,float* xpixelPerBlock, float* ypixelPerBlock, float interpolation_xgap, float interpolation_ygap, int original_image_w, int original_image_h, int resized_w, int resized_h, dim3 threadsPerBlock, dim3 numOfBlocks);