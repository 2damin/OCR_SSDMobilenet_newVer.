#pragma once
#include <iostream>
#include <cudnn.h>


void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc,
	cudnnTensorFormat_t& tensorFormat,
	cudnnDataType_t& dataType,
	int n,
	int c,
	int h,
	int w);

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
	sFilename = (std::string("test_data/") + std::string(fname));
	return;
}
