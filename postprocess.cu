
#ifndef __CUDACC_RTC__
#define __CUDACC_RTC__
#endif

#include <device_functions.h>
#include <iostream>
#include <math.h>
#include <cuda.h>
#include "postprocess.cuh"
#include "error_util.h"

using namespace std;


__global__ void remove_background(float *confData_tmp, int num_anchors, int num_classes, int image_width)
{
	int idxRow = threadIdx.x + blockDim.x*blockIdx.x;
	int idxCol = threadIdx.y + blockDim.y*blockIdx.y;
	int backgroundCol = 0;
	if (idxRow < image_width && idxCol < image_width)
	{
		for (int i = 0; i < num_anchors; i++) 
		{
			backgroundCol = image_width * i * (num_classes+1);
			confData_tmp[(idxCol+backgroundCol)*image_width + idxRow] = -1000.0;
		}
		
	}
}

__global__ void encode_locData(float *locData, int num_anchors, float* anchorShape, int box_code, int featuremap_height, int featuremap_width, int original_image_height, int original_image_width, int count_layer)
{
	__shared__ float cache[12000];
	int idxRow = threadIdx.x + blockDim.x * blockIdx.x;
	int idxCol = threadIdx.y + blockDim.y * blockIdx.y;
	//int channels = num_anchors * box_code;
	int depth = idxCol / featuremap_height;
	int idxCol_eachchannel = idxCol % featuremap_height;

	double box_xcenter = idxRow * (double)1 / featuremap_width + 0.5 * (double)1 / featuremap_width;
	double box_ycenter = idxCol_eachchannel * (double)1 / featuremap_height + 0.5 * (double)1 / featuremap_height;
	double box_width = (double)anchorShape[ num_anchors * 2 * count_layer + 2 * (depth / box_code) + 1];
	double box_height = (double)anchorShape[num_anchors * 2 * count_layer + 2 * (depth / box_code)];

	// locData= [y_center, x_center, height, width]
	if (depth % box_code == 3)
	{
		locData[idxCol * featuremap_width + idxRow] = (expf((float)(locData[idxCol * featuremap_width + idxRow] * 0.2)) * (float)box_width) * original_image_width;
	}
	else if (depth % box_code ==2)
	{
		locData[idxCol * featuremap_width + idxRow] = (expf((float)(locData[idxCol * featuremap_width + idxRow] * 0.2)) * (float)box_height) * original_image_height;
	}
	else if (depth % box_code == 1)
	{
		locData[idxCol * featuremap_width + idxRow] = ((float)(locData[idxCol * featuremap_width + idxRow] * 0.1) * (float)box_width + (float)box_xcenter) * original_image_width;
	}
	else if (depth % box_code == 0)
	{
		locData[idxCol * featuremap_width + idxRow] = ((float)(locData[idxCol * featuremap_width + idxRow] * 0.1) * (float)box_height + (float)box_ycenter) * original_image_height;
	}
}

__global__ void sum_encodedData(float* locData, int featuremap_width, int featuremap_height, int box_code)
{
	__shared__ float cache[12000];
	int one_depth = featuremap_width*featuremap_height;
	int idxCol = (threadIdx.y + blockDim.y * blockIdx.y) % featuremap_height;
	int idxRow = threadIdx.x + blockDim.x * blockIdx.x;
	int idxBox = 4 * ((threadIdx.y + blockDim.y * blockIdx.y) / featuremap_height);
	
	cache[idxBox * one_depth + idxCol * featuremap_width + idxRow] = locData[one_depth * idxBox + idxCol * featuremap_width + idxRow] - locData[one_depth *(idxBox + 2) + idxCol*featuremap_width + idxRow] * 0.5; //y_min
	cache[(1 + idxBox) * one_depth + idxCol * featuremap_width + idxRow] = locData[one_depth * (idxBox + 1) + idxCol * featuremap_width + idxRow] - locData[one_depth *(idxBox + 3) + idxCol*featuremap_width + idxRow] * 0.5; //x_min
	cache[(2 + idxBox) * one_depth + idxCol * featuremap_width + idxRow] = locData[one_depth * idxBox + idxCol * featuremap_width + idxRow] + locData[one_depth *(idxBox + 2) + idxCol*featuremap_width + idxRow] * 0.5; //y_max
	cache[(3 + idxBox) * one_depth + idxCol * featuremap_width + idxRow] = locData[one_depth * (idxBox + 1) + idxCol *  featuremap_width + idxRow] + locData[one_depth *(idxBox + 3) + idxCol*featuremap_width + idxRow] * 0.5; //x_max

	__syncthreads();

	locData[idxBox * one_depth + idxCol * featuremap_width + idxRow] = cache[idxBox * one_depth + idxCol * featuremap_width + idxRow];
	locData[(1 + idxBox) * one_depth + idxCol * featuremap_width + idxRow] = cache[(1 + idxBox) * one_depth + idxCol * featuremap_width + idxRow];
	locData[(2 + idxBox) * one_depth + idxCol * featuremap_width + idxRow] = cache[(2 + idxBox) * one_depth + idxCol * featuremap_width + idxRow];
	locData[(3 + idxBox) * one_depth + idxCol * featuremap_width + idxRow] = cache[(3 + idxBox) * one_depth + idxCol * featuremap_width + idxRow];
}

__global__ void clip(float *locData, int num_anchors, int box_code, int featuremap_height, int featuremap_width, int original_image_h, int original_image_w)
{
	int one_depth = featuremap_width*featuremap_height;
	int idxCol = (threadIdx.y + blockDim.y * blockIdx.y) % featuremap_height;
	int idxRow = threadIdx.x + blockDim.x * blockIdx.x;
	int idxBox = 4 * ((threadIdx.y + blockDim.y * blockIdx.y) / featuremap_height);

	//y_min
	locData[idxBox * one_depth + idxCol * featuremap_width + idxRow] = fmaxf((float)0, fminf(locData[idxBox * one_depth + idxCol * featuremap_width + idxRow], (float)original_image_h));
	//y_max
	locData[(idxBox + 2)* one_depth + idxCol * featuremap_width + idxRow] = fmaxf((float)0, fminf(locData[(idxBox + 2) * one_depth + idxCol * featuremap_width + idxRow], (float)original_image_h));
	//x_min
	locData[(idxBox + 1)* one_depth + idxCol * featuremap_width + idxRow] = fmaxf((float)0, fminf(locData[(idxBox + 1) * one_depth + idxCol * featuremap_width + idxRow], (float)original_image_w));
	//x_max
	locData[(idxBox + 3)* one_depth + idxCol * featuremap_width + idxRow] = fmaxf((float)0, fminf(locData[(idxBox + 3) * one_depth + idxCol * featuremap_width + idxRow], (float)original_image_w));
}

__global__ void sum_boxes(float *Data, float *Data_all, int data_startXpoint, int y_axis, int box_total)
{
	//int idxCol = threadIdx.y;
	int idxPerThread = threadIdx.x + threadIdx.y * blockDim.x;
	int idxTotal = idxPerThread + blockDim.x * blockDim.y * blockIdx.x;
	for (int i = 0; i < y_axis; i++) {
		Data_all[i * (box_total) + idxTotal + data_startXpoint] = Data[blockIdx.x * (blockDim.x * blockDim.y) * (y_axis) + i * (blockDim.x * blockDim.y) + idxPerThread];
	}
	
}

//void softmax(float* srcData, float* dstData, int anchor_num, int channel, int h, int w) {
//	cout << fixed;
//	cout.precision(7);
//	float* scoreSum = new float[anchor_num * h *w];
//	memset(scoreSum, 0, sizeof(float)*anchor_num * h *w);
//	float* srcexp = new float[channel * h * w];
//	memset(srcexp, 0, sizeof(float)*channel * h *w);
//
//	for (int k = 0; k < channel; ++k) {
//		for (int j = 0; j < h; ++j) {
//			for (int i = 0; i < w; ++i) {
//				srcexp[k* h *w + j * w + i] = expf((float)srcData[k* h *w + j * w + i]);
//				//srcexp[k* h *w + j * w + i] = srcData[k* h *w + j * w + i];
//				if (k % 9 == 0) {
//					srcexp[k* h *w + j * w + i] = 0;
//				}
//				//cout << srcexp[k* h *w + j * w + i] << "\t";
//			} //cout << endl;
//		}//cout << "=============" << endl;
//	}
//
//	for (int q = 0; q < anchor_num; ++q) {
//		for (int j = 0; j < h; ++j) {
//			for (int i = 0; i < w; ++i) {
//				for (int k = 0; k < 9; ++k) {
//					scoreSum[q * h * w + j*w + i] += srcexp[q * 9 * h * w + k* h *w + j * w + i];
//				}
//			}
//		}
//	}
//	
//	for (int q = 0; q < anchor_num; ++q) {
//		for (int k = 0; k < 9; ++k) {
//			for (int j = 0; j < h; ++j) {
//				for (int i = 0; i <w; ++i) {
//					dstData[q*9* h * w + k* h *w + j * w + i] = srcexp[q * 9 * h * w + k* h *w + j * w + i] / scoreSum[q *h * w + j * w + i];
//				}
//			}
//		}
//	}
//	
//
//
//}

void remove_background(float *confData_tmp, int num_anchors, int num_classes, int image_width, dim3 threads_per_block, dim3 num_of_blocks)
{
	remove_background << <num_of_blocks, threads_per_block >> > (confData_tmp, num_anchors, num_classes, image_width);
	return;
}

void encode_locData(float *locData, int num_anchors, float* anchorShape, int box_code, int featuremap_height, int featuremap_width, int original_image_height, int original_image_width, int count_layer, dim3 threads_per_block, dim3 num_of_blocks)
{
	encode_locData << < num_of_blocks, threads_per_block >> > (locData, num_anchors, anchorShape, box_code, featuremap_height, featuremap_width, original_image_height, original_image_width, count_layer);
	
	cudaDeviceSynchronize();

	dim3 num_of_blocks_sumData(1, num_anchors, 1);
	sum_encodedData << <num_of_blocks_sumData, threads_per_block >> > (locData, featuremap_width, featuremap_height, box_code);
	return;
}

void clip_window(float *locData, int num_anchors, int box_code, int featuremap_height, int featuremap_width, int original_image_h, int original_image_w, dim3 threads_per_block, dim3 num_of_blocks)
{
	clip << < num_of_blocks, threads_per_block >> > ( locData, num_anchors, box_code, featuremap_height, featuremap_width, original_image_h, original_image_w);
	return;
}

void sum_boxes(float *locData, float * confData, float *locData_all, float *confData_all, int class_num, int *box_featuremap_size, int *anchor_num, int box_index, int box_code, int box_total) {
	dim3 conf_blocks(anchor_num[box_index], 1, 1);
	dim3 conf_threads(box_featuremap_size[box_index], box_featuremap_size[box_index], 1);
	int data_startPoint = 0;

	for (int i = 0; i < box_index; i++)
	{
		data_startPoint += box_featuremap_size[i] * box_featuremap_size[i] * anchor_num[i];
	}
	int data_startPoint_conf = data_startPoint;
	int data_startPoint_loc = data_startPoint;

	sum_boxes << <conf_blocks, conf_threads >> > (confData, confData_all, data_startPoint_conf, class_num+1, box_total);
	cudaDeviceSynchronize();
	sum_boxes << < conf_blocks, conf_threads >> > (locData, locData_all, data_startPoint_loc, box_code, box_total);
	cudaDeviceSynchronize();
	return;
}

int descending(const void* a, const void* b)
{
	if (*(int *)a < *(int *)b)
		return 1;
	else if (*(int *)a > *(int *)b)
		return -1;
	else
		return 0;
}

int find_index(float *data, int size, float key) {
	int low, middle, high;

	low = 0;
	high = size - 1;
	while (low <= high) {
		middle = (low + high) / 2;
		if (key == data[middle]) {
			return middle;
		}
		else if (key > data[middle]) {
			high = middle - 1;
			
		}
		else { low = middle + 1; }
	}
}

float rectangleSize(float *box_offset, int index, int box_total) {
	float box_h = box_offset[2 * box_total + index] - box_offset[0 * box_total + index];
	float box_w = box_offset[3 * box_total + index] - box_offset[1 * box_total + index];

	return box_h * box_w;
}

float middleValue(float * box_offset, int index, int box_total) {
	float middle_y = (box_offset[2 * box_total + index] + box_offset[0 * box_total + index]) *0.5;
	float middle_x = (box_offset[3 * box_total + index] + box_offset[1 * box_total + index]) *0.5;

	return middle_y, middle_x;
}

float iou(float *box_offset, int index, int next_index, int box_total) {
	float box_h = 0, box_w = 0, nextBox_h = 0, nextBox_w = 0;
	float intersec_xMin = 0, intersec_xMax = 0, intersec_yMin = 0, intersec_yMax = 0;
	float intersec_w, intersec_h;

	intersec_xMin = fmaxf(box_offset[box_total * 1 + index], box_offset[box_total * 1 + next_index]);
	intersec_yMin = fmaxf(box_offset[box_total * 0 + index], box_offset[box_total * 0 + next_index]);
	intersec_xMax = fminf(box_offset[box_total * 3 + index], box_offset[box_total * 3 + next_index]);
	intersec_yMax = fminf(box_offset[box_total * 2 + index], box_offset[box_total * 2 + next_index]);

	float intersec_area = (fmaxf(0, (intersec_yMax - intersec_yMin)) * fmaxf(0, (intersec_xMax - intersec_xMin)));
	float box_area = rectangleSize(box_offset, index, box_total);
	float nextBox_area = rectangleSize(box_offset, next_index, box_total);

	return (intersec_area / (box_area + nextBox_area - intersec_area));
}

vector<int> nms(float * box_loc, vector<pair<int,int>> & address_index,
	const float & threshold, int box_total)
{
	int last;
	int i;
	vector<int> pick;
	vector<int> deleteIdxs;
	vector<pair<int,int>>::iterator iter = address_index.begin();
	// keep looping while some indexes still remain in the indexes list
	for (int k = 0; k < address_index.size() - 1; k++) {
		last = address_index.size() - 1 - k;
		i = address_index[last].first;
		
		for (int j = 0; j < last ; j++)
		{
			auto iou_result = iou(box_loc, i, address_index[j].first, box_total);
			
			if (iou_result > threshold)
			{
				deleteIdxs.push_back(last);
				break;
			}
		}
	}

	for (int k = 0; k < deleteIdxs.size(); k++) {
		iter += deleteIdxs[k];
		address_index.erase(iter);
		iter = address_index.begin();
	}

	vector<int>().swap(pick);
	vector<int>().swap(deleteIdxs);
	return pick;
}

bool compare(const pair<int, int>& a, const pair<int, int>& b) {
	return a.second < b.second;
}

vector<int> sort_by_sequence(string* result_word, float *box_loc, vector<pair<int, int>>& address_index, int box_total) {
	/*box_loc[ 0 * box_total + index] : Y_topleft
	--box_loc[ 1 * box_total + index] : X_topleft
	--box_loc[ 2 * box_total + index] : Y_bottomright
	--box_loc[ 3 * box_total + index] : X_bottomright */

	vector<pair<float, float>> box_leftTop;
	
	const float box_height = box_loc[2 * box_total + address_index[0].first] - box_loc[0 * box_total + address_index[0].first];
	float threshold_lineBreak = 0;

	//box_leftTop.first : Y_topleft, box_leftTop.second : X_topleft
	for (int i = 0; i < address_index.size(); i++) {
		box_leftTop.push_back(pair<float, float>(box_loc[0 * box_total + address_index[i].first], box_loc[1 * box_total + address_index[i].first]));
	}

	//set the threshold_linebreak
	for (int i = 0; i < address_index.size(); i++) {
		//cout << box_loc[2 * box_total + address_index[i].first] << " " << box_loc[0 * box_total + address_index[(i + 1)%address_index.size()].first] << endl;
		if (box_loc[2 * box_total + address_index[i].first] < box_loc[0 * box_total + address_index[(i + 1) % address_index.size()].first]) {
			threshold_lineBreak = (box_loc[0 * box_total + address_index[(i + 1) % address_index.size()].first] + box_loc[0 * box_total + address_index[i].first]) / 2;
			break;
		}
	}

	//sort box_leftTop by ascending(X_topleft) 
	sort(box_leftTop.begin(), box_leftTop.end(), compare);

	float last_value = box_leftTop[box_leftTop.size() - 2].second;
	float tmp[2] = { 0.0, 0.0 };
	int k = 0;
	vector<int> secondLine;
	vector<pair<float, float>>::iterator iter = box_leftTop.begin();
	for (int i = 0; i < address_index.size(); i++) {
		if (box_leftTop[i].first > threshold_lineBreak) {
			secondLine.push_back(i);
		}
	}

	//line break process
	for (int i = 0; i < secondLine.size(); i++) {
		iter += secondLine[i] - k;
		tmp[0] = box_leftTop[secondLine[i] - k].first;
		tmp[1] = box_leftTop[secondLine[i] - k].second;
		box_leftTop.erase(iter);
		box_leftTop.push_back(pair<float, float>(tmp[0], tmp[1]));
		iter = box_leftTop.begin();
		k += 1;
	}

	vector<int> sequence;
	for (int i = 0; i < box_leftTop.size(); i++) {
		//cout << box_leftTop[i].first << " , " << box_leftTop[i].second << endl;
		for (int j = 0; j < address_index.size(); j++) {
			if (box_loc[0 * box_total + address_index[j].first] == box_leftTop[i].first) {
				sequence.push_back(address_index[j].second);
			}
		}
	}
	vector<int>().swap(secondLine);
	vector<pair<float,float>>().swap(box_leftTop);

	return sequence;	
}

