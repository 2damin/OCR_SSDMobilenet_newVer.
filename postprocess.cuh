#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <algorithm>

using namespace std;

void remove_background(float *confData_tmp, int num_anchors, int num_classes, int image_width, dim3 threads_per_block, dim3 num_of_blocks);

void encode_locData(float *locData, int num_anchors, float* anchorShape, int box_code, int featuremap_height, int featuremap_width, int original_image_height, int original_image_width, int count_layer,
	dim3 threads_per_block, dim3 num_of_blocks);

void clip_window(float *locData, int num_anchors, int box_code, int featuremap_height, int featuremap_width, int original_image_h, int original_image_w, dim3 threads_per_block, dim3 num_of_blocks);

//void softmax(float* srcData, float* dstData, int anchor_num, int channel, int h, int w);

int descending(const void*, const void*);

int find_index(float *data, int size, float key);

void sum_boxes(float *locData, float * confData, float *locData_all, float *confData_all, int class_num, int *featuremapSize, int *anchor_num, int box_index, int box_code, int box_total);

float rectangleSize(float *box_offset, int index, int box_total);

float middleValue(float * box_offset, int index, int box_total);

float iou(float *box_offset, int index, int next_index, int box_total);

vector<int> nms(float *box_loc, vector<pair<int,int>> & address_index,
	const float & threshold, int box_total);

vector<int> sort_by_sequence(string* result_word, float *box_loc, vector<pair<int, int>>& address_index, int box_total);