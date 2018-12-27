#pragma once
#include <iostream>
#include <vector>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "opencv2\core\core.hpp"
#include "opencv2\opencv.hpp"
#include <sstream>
#include <cuda.h>
#include <cudnn.h>
#include "preprocess.cuh"
#include "error_util.h"

using namespace std;
using namespace cv;

void image_resize(Mat* input_image, float ** image_h, int* original_image_h, int* original_image_w, int resized_h, int resized_w); // const char* sFilename

void draw_box(const char* sFilename, float *locData, int box_address, int box_total);

void draw_allboxes(Mat* output_image, float *locData, vector<pair<int,int>> & box_address, int box_total);