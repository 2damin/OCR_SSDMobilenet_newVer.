#include "image_resizer.h"

using namespace cv;
using namespace std;

void resize_image(float* srcData, float* dstData, float* xpixelPerBlock, float* ypixelPerBlock, int* original_image_w, int* original_image_h, int resized_w, int resized_h, float xGap, float yGap)
{
	dim3 threadsPerBlock((int)xGap+1, (int)yGap+1, 1);
	dim3 numOfBlocks(*original_image_w - 1, *original_image_h - 1, 1);

	//float interpolation_xgap = (float)(resized_w - 1) / (float)(*original_image_w - 1);
	//float interpolation_ygap = (float)(resized_h - 1) / (float)(*original_image_h - 1);

	resizeCuda(srcData, dstData, xpixelPerBlock, ypixelPerBlock, xGap, yGap, *original_image_w, *original_image_h, resized_w, resized_h, threadsPerBlock, numOfBlocks);

	//float* tmpImage = new float[resized_w *resized_h];
	//memset(tmpImage, 0, sizeof(float)*resized_w*resized_h);
	//checkCudaErrors(cudaMemcpy(tmpImage, dstData, sizeof(float)*resized_w*resized_h, cudaMemcpyDeviceToHost));
	//for (int i = 0; i < resized_h; i++) {
	//	for (int j = 2; j < 3; j++) {
	//		cout << tmpImage[j + i * resized_w] << "\t";
	//	}cout << endl;
	//}
	//system("pause");
	return;
}

void image_resize(Mat* input_image, float** image_d, int* original_image_h, int* original_image_w, int resized_h, int resized_w) //const char* sFilename
{
	//Mat src = imread(sFilename);
	Mat src = *input_image;
	Mat src2;
	cvtColor(src, src, CV_RGB2BGR);
	src.convertTo(src, CV_32FC3);
	

	//for (int k = 0; k <3; k++)
	//{
	//	for (int i = 0; i < src.rows; i++) {
	//		for (int j = 0; j < src.cols; j++) {
	//			if (data_tmp[i*src.cols * 3 + j * 3 + k] > 200){
	//				cout << " j : " << j << "  " << " i : " << i << endl;
	//				printf("%0.1f \t", data_tmp[i*src.cols * 3 + j * 3 + k]); cout << endl;
	//			}
	//		} cout << endl;
	//	}
	//}system("pause");

	cvtColor(src, src, CV_BGR2GRAY);
	
	src.convertTo(src2, CV_32FC1);

	*original_image_w = src.cols;
	*original_image_h = src.rows;

	if (!src.data) {
		cout << "no image data" << endl;
		system("pause");
	}

	/*resize(src2, dst, Size(300, 300), 0, 0, INTER_LINEAR);
	uchar* data = (uchar*)dst.data;
	dst.convertTo(dst2, CV_32FC1);
	float* dstData_cv = (float*)dst2.data;
	for (int i = 0; i < 300; i++) {
		for (int j = 0; j < 300; j++) {
			cout << dstData_cv[j * 300 + i] << " ";
		}cout << endl; system("pause");
	}*/

	float* srcData = (float*)src2.data;
	float* srcData_dev = NULL;
	float wGap = (resized_w - 1)/ (float)(*original_image_w-1);
	float hGap = (resized_h - 1)/ (float)(*original_image_h-1);

	float* xpixelPerBlock = new float[(const int)*original_image_w];
	float* ypixelPerBlock = new float[(const int)*original_image_h];
	float* xpixelPerBlock_dev = NULL;
	float* ypixelPerBlock_dev = NULL;

	for (int w = 0; w < *original_image_w; ++w) {
		xpixelPerBlock[w] = (wGap * w);
	} 
	for (int h = 0; h < *original_image_h; ++h) {
		ypixelPerBlock[h] = (hGap * h);
	}
	//Corner Pixel 
	xpixelPerBlock[0] = -1;
	ypixelPerBlock[0] = -1;
	xpixelPerBlock[*original_image_w - 1] = resized_w - 1;
	ypixelPerBlock[*original_image_h - 1] = resized_h - 1;

	////test//
	//for (int i = 0; i < *original_image_w; i++) {
	//	cout << xpixelPerBlock[i] << endl;
	//}system("pause");

	checkCudaErrors(cudaMalloc(&srcData_dev, sizeof(float)*(*original_image_h)*(*original_image_w) * 1));
	checkCudaErrors(cudaMalloc(&xpixelPerBlock_dev, sizeof(float)*(*original_image_w)));
	checkCudaErrors(cudaMalloc(&ypixelPerBlock_dev, sizeof(float)*(*original_image_h)));
	checkCudaErrors(cudaMemcpy(srcData_dev, srcData, sizeof(float)*(*original_image_h)*(*original_image_w) * 1, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(xpixelPerBlock_dev, xpixelPerBlock, sizeof(float)*(*original_image_w), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(ypixelPerBlock_dev, ypixelPerBlock, sizeof(float)*(*original_image_h), cudaMemcpyHostToDevice));

	//for (int c = 0; c < src2.channels(); ++c) {
	//	for (int h = 0; h < src2.rows; ++h) {
	//		for (int w = 0; w < src2.cols; ++w) {
	//			printf("%0.2f \t", srcData[c*src2.rows*src2.cols + h*src2.cols + w]);
	//			
	//		} printf("\n");
	//	}
	//}
	//system("pause");

	resize_image(srcData_dev, *image_d, xpixelPerBlock_dev, ypixelPerBlock_dev, original_image_w, original_image_h, resized_w, resized_h, wGap, hGap);

	//uchar* data1 = (uchar*)src.data;
	//float* data = (float*)dst.data;
	//cout << dst.channels() << endl;

	//for (int c = 0; c < dst.channels(); ++c) {
	//	for (int i = 0; i < dst.cols; ++i) {
	//		for (int j = 0; j < dst.rows; ++j) {
	//			*image_h[c*dst.cols*dst.rows + i * dst.rows + j] = 2*(float(data[c*dst.cols*dst.rows + i*dst.rows + j] / float(255)))-1;
	//			//printf("%f ", test[i * src.cols + j]);
	//			//printf("%0.3f \t", data[i*dst.cols + j]);
	//		} //printf("\n ============================================================= \n"); system("pause");
	//		  //data_out += src.step;
	//	}//system("pause");
	//}
	src.release();
	src2.release();
	delete[] xpixelPerBlock;
	delete[] ypixelPerBlock;
	checkCudaErrors(cudaFree(srcData_dev));
	checkCudaErrors(cudaFree(xpixelPerBlock_dev));
	checkCudaErrors(cudaFree(ypixelPerBlock_dev));
	return;
}

void draw_box(const char* sFilename, float *locData, int box_address, int box_total)
{
	Mat src = imread(sFilename);
	Mat src_up;

	if (!src.data) {
		cout << "no image data" << endl;
		system("pause");
	}
	//resize(src, dst, Size(300, 300), 0, 0, INTER_LINEAR);
	//namedWindow("dst", WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL);
	//imshow("dst", dst);
	//waitKey(33);
	//system("pause");

	rectangle(src, Point(locData[1*box_total + box_address], locData[box_address]),
		Point(locData[3 * box_total + box_address], locData[2 * box_total + box_address]), Scalar(100, 10, 100), 1);
	//cout << locData[1 * box_total + box_address] << endl;
	
	resize(src, src_up, Size(src.rows * 5, src.cols * 5), 0, 0, INTER_LINEAR);
	namedWindow("dst", WINDOW_GUI_NORMAL);
	imshow("dst", src_up);

	waitKey(33);
	src.release();
	src_up.release();
	return;
}

void draw_allboxes(Mat* output_image, float *locData, vector<pair<int,int>> & box_address, int box_total)
{
	Mat dst = *output_image;

	if (!dst.data) {
		cout << "no image data" << endl;
		system("pause");
	}

	for (int i = 0; i < box_address.size(); i++)
	{
		rectangle(dst, Point(locData[1 * box_total + box_address[i].first], locData[box_address[i].first]),
			Point(locData[3 * box_total + box_address[i].first], locData[2 * box_total + box_address[i].first]), Scalar(10, 100, 200), 1);
		//cout << locData[1 * box_total + box_address[i]] << endl;
	}
	
	//namedWindow("dst", WINDOW_AUTOSIZE );
	//imshow("dst", dst);
	//waitKey(33);
	dst.release();
	return;
}


//int main(void)
//{
//	//string path = "6.png";
//	const char* path = "6.png";
//	float *dst = NULL;
//	float *src = new float[300 * 300 * sizeof(float)];
//	cudaMalloc(&dst, 300 * 300 * sizeof(float));
//	cudaMemset(dst, 0, 300 * 300 * sizeof(float));
//	memset(src, 0, 300 * 300 * sizeof(float));
//
//	image_resize(path, src);
//
//	cudaMemcpy(dst, src, 300*300*sizeof(float), cudaMemcpyHostToDevice);
//	cout << "이미지 로딩 완료" << endl;
//	/*for (int i = 0; i < 300; ++i) {
//		for (int j = 0; j < 300; ++j) {
//			printf("%0.1f ", src[i * 300 + j]);
//		} printf("\n");
//	}*/
//	system("pause");
//	return 0;
//}


