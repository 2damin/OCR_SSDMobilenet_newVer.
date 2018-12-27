#pragma once
#include <iostream>
#include "opencv2\opencv.hpp"
#include "mobilenet.cu"


using namespace cv;
using namespace std;

class OCR
{
public:
	
	Layer_t<float> conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17;
	Layer_t<float> box0_loc, box0_conf, box1_loc, box1_conf, box2_loc, box2_conf, box3_loc, box3_conf, box4_loc, box4_conf, box5_loc, box5_conf;
	network_t<float> ssd_mobilenet;

	void init();

	void inference(Mat* input_image, Mat* output_image, string* result_word);

	OCR();
	~OCR();
};