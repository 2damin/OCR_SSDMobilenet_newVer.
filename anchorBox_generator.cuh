#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void anchorbox_generate(float* aspect_ratio_layer0 ,float* aspect_ratio, float* scales_layer0, const float minScale, const float maxScale, const int anchorChannel, float* anchorShapeLayer0, float* anchorShapeLayer1to5, dim3 threads_per_block, dim3 num_of_blocks);
