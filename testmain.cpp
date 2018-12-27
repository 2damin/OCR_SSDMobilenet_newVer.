#include "OCR.h"

using namespace cv;

int main() {
	
	std::string image_path, result_word;
	result_word = std::string("");

	image_path = std::string("test_data/") + std::string("test_image/30.png");

	Mat src = imread(image_path.c_str(), cv::IMREAD_COLOR);
	Mat dst;
	OCR Ocr_test;
	int version = (int)cudnnGetVersion();
	int deviceId;
	cudaGetDevice(&deviceId);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceId);
	printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
	cout << "SM :" << props.multiProcessorCount << endl;
	cout << "warp : " << props.warpSize << endl;
	Ocr_test.init();
	for (int i = 0; i < 1; i++) {
		src.copyTo(dst);
		Ocr_test.inference(&src, &dst, &result_word);
		//result_word = std::string("");
	}
	
	
	cout << result_word << endl;
	imwrite("result_image/30_1218.png", dst);
	namedWindow("result", WINDOW_GUI_EXPANDED);
	imshow("result", dst);
	waitKey(0);

	dst.release();
	src.release();
	return 0;
}