#pragma once
#include <utility>
#include <time.h>
#include "dsc_conv.cuh"
#include "image_resizer.h"
#include "postprocess.cuh"
#include "anchorBox_generator.cuh"
#include "Standard_convolution.cuh"
#include "Function_others.cuh"

#define IMAGE_C 3
#define IMAGE_H 300
#define IMAGE_W 300
#define num_classes 35        //without background class
#define box_code 4            // center_x, center_y, w, h of box
#define box_total 1917       // (19 * 19) * 3 + (10 * 10 + 5 * 5 + 3 * 3 + 2 * 2 + 1 * 1) * 6   
                                    // ->  6 feature maps (1st feature map has 3 type aspect ratio of boxes and each of the others has 6 type aspect ratio of boxes) 
#define score_threshold 0.3 //box score threshold
#define EXIT_WAIVED 0

using namespace std;

const char *class_name[num_classes+1] = {"","0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "i", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };

/*----ssd paper setting-----*/
//#define minScale 0.2
//#define maxScale 0.95
//#define num_of_featuremaps 6
//#define num_of_boxChannel_per_layer 6

const float minScale = 0.1;
const float maxScale = 0.7;
const int num_of_boxChannel_per_layer = 6;
const int num_of_featuremaps = 6;

template <typename T>
struct ScaleFactorTypeMap { typedef T Type; };

typedef enum {
	FP16_HOST = 0,
	FP16_CUDA = 1,
	FP16_CUDNN = 2
} fp16Import_t;

template <class value_type>
struct Layer_t
{
	fp16Import_t fp16Import;
	int inputs, pointoutputs, outputs;
	int depth_kernel_dim, point_kernel_dim, conv_kernel_dim;
	value_type *pointData_h, *pointData_d, *depthData_h, *depthData_d, *pointBias_h, *pointBias_d, *depthBias_h, *depthBias_d, *convData_h, *convData_d;
	value_type *depthBnScale_h, *depthBnBias_h, *pointBnScale_h, *pointBnBias_h, *depthBnScale_d, *depthBnBias_d, *pointBnScale_d, *pointBnBias_d, *convBnBias_h, *convBnBias_d, *convBnScale_h, *convBnScale_d;
	value_type *depthBnMean_h, *depthBnMean_d, *depthBnVar_h, *depthBnVar_d, *pointBnMean_h, *pointBnMean_d, *pointBnVar_h, *pointBnVar_d, *convBnMean_h, *convBnMean_d, *convBnVar_h, *convBnVar_d;
	
	Layer_t() : pointData_h(NULL), pointData_d(NULL), pointBias_h(NULL), pointBias_d(NULL), depthData_h(NULL), depthData_d(NULL), depthBias_h(NULL), depthBias_d(NULL), depthBnScale_h(NULL), 
		depthBnBias_h(NULL), depthBnScale_d(NULL), depthBnBias_d(NULL), pointBnScale_h(NULL), pointBnBias_h(NULL), pointBnScale_d(NULL), pointBnBias_d(NULL), depthBnMean_h(NULL), 
		depthBnMean_d(NULL), depthBnVar_h(NULL), depthBnVar_d(NULL), pointBnMean_h(NULL), pointBnMean_d(NULL), pointBnVar_h(NULL), pointBnVar_d(NULL), convData_h(NULL), 
		convData_d(NULL), convBnBias_d(NULL), convBnBias_h(NULL), convBnVar_d(NULL), convBnVar_h(NULL), convBnMean_d(NULL), convBnMean_h(NULL), convBnScale_d(NULL), convBnScale_h(NULL),
		inputs(0), pointoutputs(0), outputs(0), depth_kernel_dim(0), point_kernel_dim(0), conv_kernel_dim(0), fp16Import(FP16_HOST) {};
	
	/********************************************************
	* box predict layer 
	* ******************************************************/
	Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights, const char* fname_bias, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
		: inputs(_inputs), pointoutputs(0), outputs(_outputs), depth_kernel_dim(0), point_kernel_dim(_kernel_dim), conv_kernel_dim(0),
		depthData_h(NULL), depthData_d(NULL), depthBias_h(NULL), depthBias_d(NULL), depthBnScale_h(NULL), depthBnBias_h(NULL), depthBnScale_d(NULL), depthBnBias_d(NULL),
		pointBnScale_h(NULL), pointBnBias_h(NULL), pointBnScale_d(NULL), pointBnBias_d(NULL), depthBnMean_h(NULL), depthBnMean_d(NULL), depthBnVar_h(NULL), 
		depthBnVar_d(NULL), pointBnMean_h(NULL), pointBnMean_d(NULL), pointBnVar_h(NULL), pointBnVar_d(NULL), convData_h(NULL), convData_d(NULL), convBnBias_d(NULL), convBnBias_h(NULL),
		convBnVar_d(NULL), convBnVar_h(NULL), convBnMean_d(NULL), convBnMean_h(NULL), convBnScale_d(NULL), convBnScale_h(NULL)
	{
		fp16Import = _fp16Import;
		std::string weights_path, bias_path;
		if (pname != NULL)
		{
			get_path(weights_path, fname_weights, pname);
			get_path(bias_path, fname_bias, pname);
		}
		else
		{
			weights_path = fname_weights; bias_path = fname_bias;
		}
		//printf("%s \n", weights_path.c_str()); 
		//printf("%s \n", fname_weights);
		readAllocInit(weights_path.c_str(), inputs * outputs * point_kernel_dim * point_kernel_dim,
			&pointData_h, &pointData_d);
		readAllocInit(bias_path.c_str(), outputs, &pointBias_h, &pointBias_d);
	}

	void init_layer_box(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights, const char* fname_bias, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
	{
		fp16Import = _fp16Import;
		std::string weights_path, bias_path;
		inputs = _inputs;
		outputs = _outputs;
		point_kernel_dim = _kernel_dim;
		if (pname != NULL)
		{
			get_path(weights_path, fname_weights, pname);
			get_path(bias_path, fname_bias, pname);
		}
		else
		{
			weights_path = fname_weights; bias_path = fname_bias;
		}
		//printf("%s \n", weights_path.c_str()); 
		//printf("%s \n", fname_weights);
		readAllocInit(weights_path.c_str(), inputs * outputs * point_kernel_dim * point_kernel_dim,
			&pointData_h, &pointData_d);
		readAllocInit(bias_path.c_str(), outputs, &pointBias_h, &pointBias_d);
		return;
	}

	/********************************************************
	* standard convolution layer
	* ******************************************************/
	Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights, const char* fname_bnScale, const char* fname_bnBias,
		const char* fname_bnMean, const char* fname_bnVar, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
		: inputs(_inputs), pointoutputs(0), outputs(_outputs), depth_kernel_dim(0), point_kernel_dim(0), conv_kernel_dim(_kernel_dim),
	    depthData_h(NULL), depthData_d(NULL), depthBias_h(NULL), depthBias_d(NULL), pointData_h(NULL), pointData_d(NULL), pointBias_h(NULL), pointBias_d(NULL),
		depthBnScale_h(NULL), depthBnBias_h(NULL), depthBnScale_d(NULL), depthBnBias_d(NULL), pointBnScale_h(NULL), pointBnBias_h(NULL), pointBnScale_d(NULL), pointBnBias_d(NULL),
		depthBnMean_h(NULL), depthBnMean_d(NULL), depthBnVar_h(NULL), depthBnVar_d(NULL), pointBnMean_h(NULL), pointBnMean_d(NULL), pointBnVar_h(NULL), pointBnVar_d(NULL)
	{
		fp16Import = _fp16Import;
		std::string weights_path, bnScale_path, bnBias_path, bnMean_path, bnVar_path; //bias_path
		if (pname != NULL)
		{
			get_path(weights_path, fname_weights, pname);
			//get_path(bias_path, fname_bias, pname);
			get_path(bnScale_path, fname_bnScale, pname);
			get_path(bnBias_path, fname_bnBias, pname);
			get_path(bnMean_path, fname_bnMean, pname);
			get_path(bnVar_path, fname_bnVar, pname);
		}
		else
		{
			weights_path = fname_weights; //bias_path = fname_bias;
			bnScale_path = fname_bnScale; bnBias_path = fname_bnBias;
			bnMean_path = fname_bnMean; bnVar_path = fname_bnVar;
		}

		readAllocInit(weights_path.c_str(), inputs * outputs * conv_kernel_dim * conv_kernel_dim,
			&convData_h, &convData_d);
		//readAllocInit(bias_path.c_str(), outputs, &pointBias_h, &pointBias_d);
		readAllocInit(bnScale_path.c_str(), outputs, &convBnScale_h, &convBnScale_d);
		readAllocInit(bnBias_path.c_str(), outputs, &convBnBias_h, &convBnBias_d);
		readAllocInit(bnMean_path.c_str(), outputs, &convBnMean_h, &convBnMean_d);
		readAllocInit(bnVar_path.c_str(), outputs, &convBnVar_h, &convBnVar_d);
	}

	void init_layer_standard(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights, const char* fname_bnScale, const char* fname_bnBias,
		const char* fname_bnMean, const char* fname_bnVar, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
	{
		fp16Import = _fp16Import;
		std::string weights_path, bnScale_path, bnBias_path, bnMean_path, bnVar_path; //bias_path
		conv_kernel_dim = _kernel_dim;
		outputs = _outputs;
		inputs = _inputs;
		
		if (pname != NULL)
		{
			get_path(weights_path, fname_weights, pname);
			//get_path(bias_path, fname_bias, pname);
			get_path(bnScale_path, fname_bnScale, pname);
			get_path(bnBias_path, fname_bnBias, pname);
			get_path(bnMean_path, fname_bnMean, pname);
			get_path(bnVar_path, fname_bnVar, pname);
		}
		else
		{
			weights_path = fname_weights; //bias_path = fname_bias;
			bnScale_path = fname_bnScale; bnBias_path = fname_bnBias;
			bnMean_path = fname_bnMean; bnVar_path = fname_bnVar;
		}

		readAllocInit(weights_path.c_str(), inputs * outputs * conv_kernel_dim * conv_kernel_dim,
			&convData_h, &convData_d);
		//readAllocInit(bias_path.c_str(), outputs, &pointBias_h, &pointBias_d);
		readAllocInit(bnScale_path.c_str(), outputs, &convBnScale_h, &convBnScale_d);
		readAllocInit(bnBias_path.c_str(), outputs, &convBnBias_h, &convBnBias_d);
		readAllocInit(bnMean_path.c_str(), outputs, &convBnMean_h, &convBnMean_d);
		readAllocInit(bnVar_path.c_str(), outputs, &convBnVar_h, &convBnVar_d);
		return;
	}

	/********************************************************
	* DSC layer
	* ******************************************************/
	Layer_t(int _inputs, int _outputs, int _depth_kernel_dim, int _point_kernel_dim, const char* fname_depthWeights, const char* fname_pointWeights, 
		const char* fname_depthBnScale, const char* fname_depthBnBias, const char* fname_pointBnScale, const char* fname_pointBnBias , 
		const char* fname_depthBnMean, const char* fname_depthBnVar, const char* fname_pointBnMean, const char* fname_pointBnVar,
		const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
		: inputs(_inputs), pointoutputs(0), outputs(_outputs), depth_kernel_dim(_depth_kernel_dim), point_kernel_dim(_point_kernel_dim), conv_kernel_dim(0), depthBias_h(NULL), depthBias_d(NULL),
		pointBias_h(NULL), pointBias_d(NULL), convData_h(NULL), convData_d(NULL), convBnMean_h(NULL), convBnMean_d(NULL), convBnVar_d(NULL), convBnVar_h(NULL), convBnBias_d(NULL),
		convBnBias_h(NULL), convBnScale_d(NULL), convBnScale_h(NULL)
	{
		fp16Import = _fp16Import;
		std::string depthWeights_path, pointWeights_path, depthBnScale_path, depthBnBias_path, pointBnScale_path, pointBnBias_path;
		std::string depthBnMean_path, depthBnVar_path, pointBnMean_path, pointBnVar_path;
		if (pname != NULL)
		{
			get_path(depthWeights_path, fname_depthWeights, pname);
			get_path(pointWeights_path, fname_pointWeights, pname);
			get_path(depthBnScale_path, fname_depthBnScale, pname);
			get_path(depthBnBias_path, fname_depthBnBias, pname);
			get_path(pointBnScale_path, fname_pointBnScale, pname);
			get_path(pointBnBias_path, fname_pointBnBias, pname);
			get_path(depthBnMean_path, fname_depthBnMean, pname);
			get_path(depthBnVar_path, fname_depthBnVar, pname);
			get_path(pointBnMean_path, fname_pointBnMean, pname);
			get_path(pointBnVar_path, fname_pointBnVar, pname);
		}
		else
		{
			depthWeights_path = fname_depthWeights; pointWeights_path = fname_pointWeights;
			depthBnScale_path = fname_depthBnScale; depthBnBias_path = fname_depthBnBias;
			pointBnScale_path = fname_pointBnScale; pointBnBias_path = fname_pointBnBias;
			depthBnMean_path = fname_depthBnMean; depthBnVar_path = fname_depthBnVar;
			pointBnMean_path = fname_pointBnMean; pointBnVar_path = fname_pointBnVar;
		}
		
		readAllocInit(depthWeights_path.c_str(), inputs * depth_kernel_dim * depth_kernel_dim,
			&depthData_h, &depthData_d);
		readAllocInit(pointWeights_path.c_str(), inputs * outputs * point_kernel_dim * point_kernel_dim,
			&pointData_h, &pointData_d);
		readAllocInit(depthBnScale_path.c_str(), inputs, &depthBnScale_h, &depthBnScale_d);
		readAllocInit(depthBnBias_path.c_str(), inputs, &depthBnBias_h, &depthBnBias_d);
		readAllocInit(pointBnScale_path.c_str(), outputs, &pointBnScale_h, &pointBnScale_d);
		readAllocInit(pointBnBias_path.c_str(), outputs, &pointBnBias_h, &pointBnBias_d);
		readAllocInit(depthBnMean_path.c_str(), inputs, &depthBnMean_h, &depthBnMean_d);
		readAllocInit(depthBnVar_path.c_str(), inputs, &depthBnVar_h, &depthBnVar_d);
		readAllocInit(pointBnMean_path.c_str(), outputs, &pointBnMean_h, &pointBnMean_d);
		readAllocInit(pointBnVar_path.c_str(), outputs, &pointBnVar_h, &pointBnVar_d);
	}

	void init_layer_dsc(int _inputs, int _outputs, int _depth_kernel_dim, int _point_kernel_dim, const char* fname_depthWeights, const char* fname_pointWeights,
		const char* fname_depthBnScale, const char* fname_depthBnBias, const char* fname_pointBnScale, const char* fname_pointBnBias,
		const char* fname_depthBnMean, const char* fname_depthBnVar, const char* fname_pointBnMean, const char* fname_pointBnVar,
		const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
	{
		fp16Import = _fp16Import;
		std::string depthWeights_path, pointWeights_path, depthBnScale_path, depthBnBias_path, pointBnScale_path, pointBnBias_path;
		std::string depthBnMean_path, depthBnVar_path, pointBnMean_path, pointBnVar_path;
		inputs = _inputs;
		outputs = _outputs;
		depth_kernel_dim = _depth_kernel_dim;
		point_kernel_dim = _point_kernel_dim;
		if (pname != NULL)
		{
			get_path(depthWeights_path, fname_depthWeights, pname);
			get_path(pointWeights_path, fname_pointWeights, pname);
			get_path(depthBnScale_path, fname_depthBnScale, pname);
			get_path(depthBnBias_path, fname_depthBnBias, pname);
			get_path(pointBnScale_path, fname_pointBnScale, pname);
			get_path(pointBnBias_path, fname_pointBnBias, pname);
			get_path(depthBnMean_path, fname_depthBnMean, pname);
			get_path(depthBnVar_path, fname_depthBnVar, pname);
			get_path(pointBnMean_path, fname_pointBnMean, pname);
			get_path(pointBnVar_path, fname_pointBnVar, pname);
		}
		else
		{
			depthWeights_path = fname_depthWeights; pointWeights_path = fname_pointWeights;
			depthBnScale_path = fname_depthBnScale; depthBnBias_path = fname_depthBnBias;
			pointBnScale_path = fname_pointBnScale; pointBnBias_path = fname_pointBnBias;
			depthBnMean_path = fname_depthBnMean; depthBnVar_path = fname_depthBnVar;
			pointBnMean_path = fname_pointBnMean; pointBnVar_path = fname_pointBnVar;
		}

		readAllocInit(depthWeights_path.c_str(), inputs * depth_kernel_dim * depth_kernel_dim,
			&depthData_h, &depthData_d);
		readAllocInit(pointWeights_path.c_str(), inputs * outputs * point_kernel_dim * point_kernel_dim,
			&pointData_h, &pointData_d);
		readAllocInit(depthBnScale_path.c_str(), inputs, &depthBnScale_h, &depthBnScale_d);
		readAllocInit(depthBnBias_path.c_str(), inputs, &depthBnBias_h, &depthBnBias_d);
		readAllocInit(pointBnScale_path.c_str(), outputs, &pointBnScale_h, &pointBnScale_d);
		readAllocInit(pointBnBias_path.c_str(), outputs, &pointBnBias_h, &pointBnBias_d);
		readAllocInit(depthBnMean_path.c_str(), inputs, &depthBnMean_h, &depthBnMean_d);
		readAllocInit(depthBnVar_path.c_str(), inputs, &depthBnVar_h, &depthBnVar_d);
		readAllocInit(pointBnMean_path.c_str(), outputs, &pointBnMean_h, &pointBnMean_d);
		readAllocInit(pointBnVar_path.c_str(), outputs, &pointBnVar_h, &pointBnVar_d);
		return;
	}

	/********************************************************
	* Extra layer
	* ******************************************************/
	Layer_t(int _inputs, int _pointouts, int _outputs, int _conv_kernel_dim, int _point_kernel_dim, const char* fname_convWeights, const char* fname_pointWeights,
		const char* fname_convBnScale, const char* fname_convBnBias, const char* fname_pointBnScale, const char* fname_pointBnBias,
		const char* fname_convBnMean, const char* fname_convBnVar, const char* fname_pointBnMean, const char* fname_pointBnVar,
		const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
		: inputs(_inputs), pointoutputs(_pointouts), outputs(_outputs), conv_kernel_dim(_conv_kernel_dim), point_kernel_dim(_point_kernel_dim), depth_kernel_dim(0),
		depthData_h(NULL), depthData_d(NULL), depthBias_h(NULL), depthBias_d(NULL), pointBias_d(NULL), pointBias_h(NULL),
		depthBnScale_h(NULL), depthBnBias_h(NULL), depthBnScale_d(NULL), depthBnBias_d(NULL), depthBnMean_h(NULL), depthBnMean_d(NULL), depthBnVar_h(NULL), depthBnVar_d(NULL)
	{
		fp16Import = _fp16Import;
		std::string convWeights_path, pointWeights_path, convBnScale_path, convBnBias_path, pointBnScale_path, pointBnBias_path;
		std::string convBnMean_path, convBnVar_path, pointBnMean_path, pointBnVar_path;
		if (pname != NULL)
		{
			get_path(convWeights_path, fname_convWeights, pname);
			get_path(pointWeights_path, fname_pointWeights, pname);
			get_path(convBnScale_path, fname_convBnScale, pname);
			get_path(convBnBias_path, fname_convBnBias, pname);
			get_path(pointBnScale_path, fname_pointBnScale, pname);
			get_path(pointBnBias_path, fname_pointBnBias, pname);
			get_path(convBnMean_path, fname_convBnMean, pname);
			get_path(convBnVar_path, fname_convBnVar, pname);
			get_path(pointBnMean_path, fname_pointBnMean, pname);
			get_path(pointBnVar_path, fname_pointBnVar, pname);
		}
		else
		{
			convWeights_path = fname_convWeights; pointWeights_path = fname_pointWeights;
			convBnScale_path = fname_convBnScale; convBnBias_path = fname_convBnBias;
			pointBnScale_path = fname_pointBnScale; pointBnBias_path = fname_pointBnBias;
			convBnMean_path = fname_convBnMean; convBnVar_path = fname_convBnVar;
			pointBnMean_path = fname_pointBnMean; pointBnVar_path = fname_pointBnVar;
		}

		readAllocInit(convWeights_path.c_str(), pointoutputs * outputs * conv_kernel_dim * conv_kernel_dim,
			&convData_h, &convData_d);
		readAllocInit(pointWeights_path.c_str(), inputs * pointoutputs * point_kernel_dim * point_kernel_dim,
			&pointData_h, &pointData_d);
		readAllocInit(convBnScale_path.c_str(), outputs, &convBnScale_h, &convBnScale_d);
		readAllocInit(convBnBias_path.c_str(), outputs, &convBnBias_h, &convBnBias_d);
		readAllocInit(pointBnScale_path.c_str(), pointoutputs, &pointBnScale_h, &pointBnScale_d);
		readAllocInit(pointBnBias_path.c_str(), pointoutputs, &pointBnBias_h, &pointBnBias_d);
		readAllocInit(convBnMean_path.c_str(), outputs, &convBnMean_h, &convBnMean_d);
		readAllocInit(convBnVar_path.c_str(), outputs, &convBnVar_h, &convBnVar_d);
		readAllocInit(pointBnMean_path.c_str(), pointoutputs, &pointBnMean_h, &pointBnMean_d);
		readAllocInit(pointBnVar_path.c_str(), pointoutputs, &pointBnVar_h, &pointBnVar_d);

	}

	void init_layer_extra(int _inputs, int _pointouts, int _outputs, int _conv_kernel_dim, int _point_kernel_dim, const char* fname_convWeights, const char* fname_pointWeights,
		const char* fname_convBnScale, const char* fname_convBnBias, const char* fname_pointBnScale, const char* fname_pointBnBias,
		const char* fname_convBnMean, const char* fname_convBnVar, const char* fname_pointBnMean, const char* fname_pointBnVar,
		const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
	{
		fp16Import = _fp16Import;
		std::string convWeights_path, pointWeights_path, convBnScale_path, convBnBias_path, pointBnScale_path, pointBnBias_path;
		std::string convBnMean_path, convBnVar_path, pointBnMean_path, pointBnVar_path;
		inputs = _inputs;
		pointoutputs = _pointouts;
		outputs = _outputs;
		conv_kernel_dim = _conv_kernel_dim;
		point_kernel_dim = _point_kernel_dim;

		if (pname != NULL)
		{
			get_path(convWeights_path, fname_convWeights, pname);
			get_path(pointWeights_path, fname_pointWeights, pname);
			get_path(convBnScale_path, fname_convBnScale, pname);
			get_path(convBnBias_path, fname_convBnBias, pname);
			get_path(pointBnScale_path, fname_pointBnScale, pname);
			get_path(pointBnBias_path, fname_pointBnBias, pname);
			get_path(convBnMean_path, fname_convBnMean, pname);
			get_path(convBnVar_path, fname_convBnVar, pname);
			get_path(pointBnMean_path, fname_pointBnMean, pname);
			get_path(pointBnVar_path, fname_pointBnVar, pname);
		}
		else
		{
			convWeights_path = fname_convWeights; pointWeights_path = fname_pointWeights;
			convBnScale_path = fname_convBnScale; convBnBias_path = fname_convBnBias;
			pointBnScale_path = fname_pointBnScale; pointBnBias_path = fname_pointBnBias;
			convBnMean_path = fname_convBnMean; convBnVar_path = fname_convBnVar;
			pointBnMean_path = fname_pointBnMean; pointBnVar_path = fname_pointBnVar;
		}

		readAllocInit(convWeights_path.c_str(), pointoutputs * outputs * conv_kernel_dim * conv_kernel_dim,
			&convData_h, &convData_d);
		readAllocInit(pointWeights_path.c_str(), inputs * pointoutputs * point_kernel_dim * point_kernel_dim,
			&pointData_h, &pointData_d);
		readAllocInit(convBnScale_path.c_str(), outputs, &convBnScale_h, &convBnScale_d);
		readAllocInit(convBnBias_path.c_str(), outputs, &convBnBias_h, &convBnBias_d);
		readAllocInit(pointBnScale_path.c_str(), pointoutputs, &pointBnScale_h, &pointBnScale_d);
		readAllocInit(pointBnBias_path.c_str(), pointoutputs, &pointBnBias_h, &pointBnBias_d);
		readAllocInit(convBnMean_path.c_str(), outputs, &convBnMean_h, &convBnMean_d);
		readAllocInit(convBnVar_path.c_str(), outputs, &convBnVar_h, &convBnVar_d);
		readAllocInit(pointBnMean_path.c_str(), pointoutputs, &pointBnMean_h, &pointBnMean_d);
		readAllocInit(pointBnVar_path.c_str(), pointoutputs, &pointBnVar_h, &pointBnVar_d);
		return;
	}

	~Layer_t()
	{
		if (pointData_h != NULL) delete[] pointData_h;
		if (depthData_h != NULL) delete[] depthData_h;
		if (convData_h != NULL) delete[] convData_h;
		if (pointData_d != NULL) checkCudaErrors(cudaFree(pointData_d));
		if (depthData_d != NULL) checkCudaErrors(cudaFree(depthData_d));
		if (convData_d != NULL) checkCudaErrors(cudaFree(convData_d));
		if (depthBias_h != NULL) delete[] depthBias_h;
		if (pointBias_h != NULL) delete[] pointBias_h;
		if (depthBias_d != NULL) checkCudaErrors(cudaFree(depthBias_d));
		if (pointBias_d != NULL) checkCudaErrors(cudaFree(pointBias_d));
		if (depthBnScale_h != NULL) delete[] depthBnScale_h;
		if (depthBnBias_h != NULL) delete[] depthBnBias_h;
		if (pointBnScale_h != NULL) delete[] pointBnScale_h;
		if (pointBnBias_h != NULL) delete[] pointBnBias_h;
		if (convBnScale_h != NULL) delete[] convBnScale_h;
		if (convBnBias_h != NULL) delete[] convBnBias_h;
		if (depthBnScale_d != NULL) checkCudaErrors(cudaFree(depthBnScale_d));
		if (depthBnBias_d != NULL) checkCudaErrors(cudaFree(depthBnBias_d));
		if (pointBnScale_d != NULL) checkCudaErrors(cudaFree(pointBnScale_d));
		if (pointBnBias_d != NULL) checkCudaErrors(cudaFree(pointBnBias_d));
		if (convBnScale_d != NULL) checkCudaErrors(cudaFree(convBnScale_d));
		if (convBnBias_d != NULL) checkCudaErrors(cudaFree(convBnBias_d));
		if (depthBnMean_h != NULL) delete[] depthBnMean_h;
		if (depthBnVar_h != NULL) delete[] depthBnVar_h;
		if (pointBnMean_h != NULL) delete[] pointBnMean_h;
		if (pointBnVar_h != NULL) delete[] pointBnVar_h;
		if (convBnMean_h != NULL) delete[] convBnMean_h;
		if (convBnVar_h != NULL) delete[] convBnVar_h;
		if (depthBnMean_d != NULL) checkCudaErrors(cudaFree(depthBnMean_d));
		if (depthBnVar_d != NULL) checkCudaErrors(cudaFree(depthBnVar_d));
		if (pointBnMean_d != NULL) checkCudaErrors(cudaFree(pointBnMean_d));
		if (pointBnVar_d != NULL) checkCudaErrors(cudaFree(pointBnVar_d));
		if (convBnMean_d != NULL) checkCudaErrors(cudaFree(convBnMean_d));
		if (convBnVar_d != NULL) checkCudaErrors(cudaFree(convBnVar_d));
	};

private:
	void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
	{
		readAllocMemcpy<value_type>(fname, size, data_h, data_d);
		return;
	}
	template <class value_type>
	void readBinaryFile(const char* fname, int size, value_type* data_h)
	{
		std::ifstream dataFile(fname, std::ios::in | std::ios::binary);
		std::stringstream error_s;
		if (!dataFile)
		{
			error_s << "Error opening file " << fname;
			FatalError(error_s.str());
		}
		// we assume the data stored is always in float precision
		float* data_tmp = new float[size];
		int size_b = size * sizeof(float);
		if (!dataFile.read((char*)data_tmp, size_b))
		{
			error_s << "Error reading file " << fname;
			FatalError(error_s.str());
		}
		// conversion
		//Convert<value_type> fromReal;
		for (int i = 0; i < size; i++)
		{
			data_h[i] = data_tmp[i];
			//printf("%f \n", data_tmp[i]);
		}// system("pause");

		delete[] data_tmp;
		return;
	}

	template <class value_type>
	void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d)
	{
		*data_h = new value_type[size];

		readBinaryFile<value_type>(fname, size, *data_h);


		int size_b = size * sizeof(value_type);
		checkCudaErrors(cudaMalloc(data_d, size_b));
		checkCudaErrors(cudaMemcpy(*data_d, *data_h,
			size_b,
			cudaMemcpyHostToDevice));
		//delete[] *data_h;
		return;
	}
};

// demonstrate different ways of setting tensor descriptor
//#define SIMPLE_TENSOR_DESCRIPTOR
#define ND_TENSOR_DESCRIPTOR
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc,
	cudnnTensorFormat_t& tensorFormat,
	cudnnDataType_t& dataType,
	int n,
	int c,
	int h,
	int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
	checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h,
		w));
#elif defined(ND_TENSOR_DESCRIPTOR)
	const int nDims = 4;
	int dimA[nDims] = { n,c,h,w };
	int strideA[nDims] = { c*h*w, h*w, w, 1 };
	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc,
		dataType,
		4,
		dimA,
		strideA));
#else
	checkCUDNN(cudnnSetTensor4dDescriptorEx(tensorDesc,
		dataType,
		n, c,
		h, w,
		c*h*w, h*w, w, 1));
#endif
	return;
}

template <class value_type>
class network_t
{
	typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
	int convAlgorithm;
	cudnnDataType_t dataType;
	cudnnTensorFormat_t tensorFormat;
	cudnnHandle_t cudnnHandle;
	cudnnTensorDescriptor_t srcConvTensorDesc, dstConvTensorDesc, dstTensorDesc, srcTensorDesc, bnTensorDesc, activTensorDesc;
	cudnnFilterDescriptor_t filterDesc;
	cudnnConvolutionDescriptor_t convDesc;
	//cudnnPoolingDescriptor_t     poolingDesc;
	cudnnActivationDescriptor_t  activDesc, activBoxDesc;
	//cudnnConvolutionFwdAlgo_t algo;
	//size_t sizeInBytes = 0;
	//void* workSpace = NULL;
	double epsilon = 0.001;


	void createHandles()
	{
		checkCUDNN(cudnnCreate(&cudnnHandle));
		checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&srcConvTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&activTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstConvTensorDesc));
		checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
		checkCUDNN(cudnnCreateActivationDescriptor(&activDesc));
		checkCUDNN(cudnnCreateActivationDescriptor(&activBoxDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&bnTensorDesc));
		return;
	}
	void destroyHandles()
	{
		checkCUDNN(cudnnDestroyActivationDescriptor(activDesc));
		checkCUDNN(cudnnDestroyActivationDescriptor(activBoxDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(activTensorDesc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
		checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(srcConvTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(dstConvTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(bnTensorDesc));
		checkCUDNN(cudnnDestroy(cudnnHandle));
		return;
	}
public:
	network_t()
	{
		convAlgorithm = -1;
		switch (sizeof(value_type))
		{
		case 2: dataType = CUDNN_DATA_HALF; break;
		case 4: dataType = CUDNN_DATA_FLOAT; break;
		case 8: dataType = CUDNN_DATA_DOUBLE; break;
		default: FatalError("Unsupported data type");
		}
		tensorFormat = CUDNN_TENSOR_NCHW;
		createHandles();
	};
	~network_t()
	{
		/*if (sizeInBytes != 0)
		{
			checkCudaErrors(cudaFree(workSpace));
		}*/
		destroyHandles();
		//cudaDeviceReset();
	}

	void resize(int size, value_type **data)
	{
		if (*data != NULL)
		{
			checkCudaErrors(cudaFree(*data));
		}
		checkCudaErrors(cudaMalloc(data, size * sizeof(value_type)));
		checkCudaErrors(cudaMemset(*data, 0, size * sizeof(value_type)));
		return;
	}
	void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
	{
		convAlgorithm = (int)algo;
		return;
	}

	//void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type* data)
	//{
	//	setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);
	//	scaling_type alpha = scaling_type(1);
	//	scaling_type beta = scaling_type(1);
	//	checkCUDNN(cudnnAddTensor(cudnnHandle,
	//		&alpha, biasTensorDesc,
	//		layer.pointBias_d,
	//		&beta,
	//		dstTensorDesc,
	//		data));
	//}

	void depthSetting(int n, int c, int h, int w, int col, int padding, value_type** dstData, value_type** tmp)
	{
		int tmpsize = (col + padding) * (col + padding) * c;
		resize(tmpsize, tmp);
		resize((n * c * h * w), dstData);
		return;
	}
	
	void pointSetting(int n, int c, int h, int w, value_type** dstData)
	{
		resize(n * c * h * w, dstData);
		return;
	}

	void anchorBox_generator(value_type* aspect_ratio_layer0 ,value_type* aspect_ratio, value_type** anchorShape0, value_type** anchorShape1to5)
	{
		value_type* scales_layer0 = new value_type[3];
		/*---paper setting---*/
		//aspect_ratio[0] = 1.0;
		//aspect_ratio[1] = 2.0;
		//aspect_ratio[2] = 0.5;
		//aspect_ratio[3] = 3.0;
		//aspect_ratio[4] = 0.3333333;
		//aspect_ratio[5] = 1.0;
		//scales_layer0[0] = 0.1;
		//scales_layer0[1] = 0.2;
		//scales_layer0[2] = 0.2;
		//aspect_ratio_layer0[0] = 1.0;
		//aspect_ratio_layer0[1] = 1.5;
		//aspect_ratio_layer0[2] = 0.3333;

		/*---modified setting---*/
		aspect_ratio[0] = 1.0;
		aspect_ratio[1] = 1.5;
		aspect_ratio[2] = 0.3333;
		aspect_ratio[3] = 0.75;
		aspect_ratio[4] = 0.5;
		aspect_ratio[5] = 1.0;
		scales_layer0[0] = 0.1;
		scales_layer0[1] = 0.1;
		scales_layer0[2] = 0.1;
		aspect_ratio_layer0[0] = 1.0;
		aspect_ratio_layer0[1] = 2.0;
		aspect_ratio_layer0[2] = 0.5;

		dim3 threads_per_block(32, 32, 1);
		dim3 num_of_blocks(1, 1, 1);

		checkCudaErrors(cudaMalloc(anchorShape0, sizeof(value_type) * (num_of_boxChannel_per_layer)));
		checkCudaErrors(cudaMemset(*anchorShape0, 0, sizeof(value_type) *  (num_of_boxChannel_per_layer)));

		checkCudaErrors(cudaMalloc(anchorShape1to5, sizeof(value_type) * (num_of_boxChannel_per_layer * 2 * (num_of_featuremaps-1))));
		checkCudaErrors(cudaMemset(*anchorShape1to5, 0, sizeof(value_type) *  (num_of_boxChannel_per_layer * 2 * (num_of_featuremaps - 1))));

		value_type* aspect_ratio_dev = NULL;
		checkCudaErrors(cudaMalloc(&aspect_ratio_dev, sizeof(value_type) * num_of_boxChannel_per_layer));
		checkCudaErrors(cudaMemcpy(aspect_ratio_dev, aspect_ratio, sizeof(value_type) * num_of_boxChannel_per_layer, cudaMemcpyHostToDevice));

		value_type* aspect_ratio_layer0_dev = NULL;
		checkCudaErrors(cudaMalloc(&aspect_ratio_layer0_dev, sizeof(value_type) * num_of_boxChannel_per_layer/2));
		checkCudaErrors(cudaMemcpy(aspect_ratio_layer0_dev, aspect_ratio_layer0, sizeof(value_type) * num_of_boxChannel_per_layer/2, cudaMemcpyHostToDevice));
		
		value_type* scales_layer0_dev = NULL;
		checkCudaErrors(cudaMalloc(&scales_layer0_dev, sizeof(value_type) * num_of_boxChannel_per_layer / 2));
		checkCudaErrors(cudaMemcpy(scales_layer0_dev, scales_layer0, sizeof(value_type)* num_of_boxChannel_per_layer / 2, cudaMemcpyHostToDevice));
		
		anchorbox_generate(aspect_ratio_layer0_dev ,aspect_ratio_dev, scales_layer0_dev, minScale, maxScale, num_of_boxChannel_per_layer, *anchorShape0, *anchorShape1to5, threads_per_block, num_of_blocks);

		delete[] scales_layer0;
		checkCudaErrors(cudaFree(aspect_ratio_dev));
		checkCudaErrors(cudaFree(aspect_ratio_layer0_dev));
		checkCudaErrors(cudaFree(scales_layer0_dev));
		return;
	}

	void convolutionSetting(int *n, int *c, int *h, int *w, const Layer_t<value_type>& conv, int padding, int stride, int *tensorOuputDimA, const int tensorDims, value_type** dstData)
	{
		////-----convolution pre-processing------////
		setTensorDesc(srcConvTensorDesc, tensorFormat, dataType, *n, *c, *h, *w);

		tensorOuputDimA[0] = *n;
		tensorOuputDimA[1] = *c;
		tensorOuputDimA[2] = *h;
		tensorOuputDimA[3] = *w;
		const int filterDimA[4] = { conv.outputs, *c, conv.conv_kernel_dim, conv.conv_kernel_dim };
		
		checkCUDNN(cudnnSetFilterNdDescriptor(filterDesc,
			dataType,
			CUDNN_TENSOR_NCHW,
			tensorDims,
			filterDimA));

		const int convDims = 2;
		int padA[convDims] = { padding, padding };
		int filterStrideA[convDims] = { stride, stride };
		int upscaleA[convDims] = { 1,1 };
		cudnnDataType_t  convDataType = dataType;
		if (dataType == CUDNN_DATA_HALF) {
			convDataType = CUDNN_DATA_FLOAT; //Math are done in FP32 when tensor are in FP16
		}
		checkCUDNN(cudnnSetConvolutionNdDescriptor(convDesc,
			convDims,
			padA,
			filterStrideA,
			upscaleA,
			CUDNN_CROSS_CORRELATION,
			convDataType));
		// find dimension of convolution output
		checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convDesc,
			srcConvTensorDesc,
			filterDesc,
			tensorDims,
			tensorOuputDimA));
		*n = tensorOuputDimA[0]; *c = tensorOuputDimA[1];
		*h = tensorOuputDimA[2]; *w = tensorOuputDimA[3];

		//setTensorDesc(dstConvTensorDesc, tensorFormat, dataType, *n, *c, *h, *w);

		//checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
		//	srcConvTensorDesc,
		//	filterDesc,
		//	convDesc,
		//	dstConvTensorDesc,
		//	CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		//	0,
		//	&algo
		//));

		resize((*n)*(*c)*(*h)*(*w), dstData);

		//checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		//	srcConvTensorDesc,
		//	filterDesc,
		//	convDesc,
		//	dstConvTensorDesc,
		//	algo,
		//	&sizeInBytes));
		//if (sizeInBytes != 0)
		//{
		//	checkCudaErrors(cudaMalloc(&workSpace, sizeInBytes));
		//}
		return;
	}

	/*--------cuDNN ConvolutionForward--------*/

	//void convoluteForward(const Layer_t<value_type>& conv, int n, int c, int h, int w, cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t sizeInBytes, value_type* srcData, value_type** dstData)
	//{
	//	scaling_type alpha = scaling_type(1);
	//	scaling_type beta = scaling_type(0);
	//	checkCUDNN(cudnnConvolutionForward(cudnnHandle,
	//		&alpha,
	//		srcConvTensorDesc,
	//		srcData,
	//		filterDesc,
	//		conv.convData_d,
	//		convDesc,
	//		algo,
	//		workSpace,
	//		sizeInBytes,
	//		&beta,
	//		dstConvTensorDesc,
	//		*dstData));
	//	//addBias(dstConvTensorDesc, conv, c, *dstData);
	//	if (sizeInBytes != 0)
	//	{
	//		checkCudaErrors(cudaFree(workSpace));
	//	}
	//}

	void batchNorm_depth(int n, int c, int h, int w ,const Layer_t<value_type>& conv , cudnnTensorDescriptor_t tensorDesc ,value_type* srcData, value_type** dstData )
	{
		setTensorDesc(tensorDesc, tensorFormat, dataType, n, c, h, w);
		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		resize(n*c*h*w, dstData);
		
		checkCUDNN(cudnnDeriveBNTensorDescriptor(bnTensorDesc, tensorDesc, CUDNN_BATCHNORM_SPATIAL));
		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle,
			CUDNN_BATCHNORM_SPATIAL,
			&alpha,
			&beta,
			tensorDesc,
			srcData,
			tensorDesc,
			*dstData,
			bnTensorDesc,
			conv.depthBnScale_d,
			conv.depthBnBias_d,
			conv.depthBnMean_d,
			conv.depthBnVar_d,
			epsilon));
		return;
	}

	void batchNorm_point(int n, int c, int h, int w, const Layer_t<value_type>& conv, cudnnTensorDescriptor_t tensorDesc, value_type* srcData, value_type** dstData)
	{
		setTensorDesc(tensorDesc, tensorFormat, dataType, n, c, h, w);
		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		resize(n*c*h*w, dstData);

		checkCUDNN(cudnnDeriveBNTensorDescriptor(bnTensorDesc, tensorDesc, CUDNN_BATCHNORM_SPATIAL));
		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle,
			CUDNN_BATCHNORM_SPATIAL,
			&alpha,
			&beta,
			tensorDesc,
			srcData,
			tensorDesc,
			*dstData,
			bnTensorDesc,
			conv.pointBnScale_d,
			conv.pointBnBias_d,
			conv.pointBnMean_d,
			conv.pointBnVar_d,
			epsilon));
		return;
	}

	void batchNorm_conv(int n, int c, int h, int w, const Layer_t<value_type>& conv, cudnnTensorDescriptor_t tensorDesc, value_type* srcData, value_type** dstData)
	{
		setTensorDesc(tensorDesc, tensorFormat, dataType, n, c, h, w);
		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		resize(n*c*h*w, dstData);

		checkCUDNN(cudnnDeriveBNTensorDescriptor(bnTensorDesc, tensorDesc, CUDNN_BATCHNORM_SPATIAL));
		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle,
			CUDNN_BATCHNORM_SPATIAL,
			&alpha,
			&beta,
			tensorDesc,
			srcData,
			tensorDesc,
			*dstData,
			bnTensorDesc,
			conv.convBnScale_d,
			conv.convBnBias_d,
			conv.convBnMean_d,
			conv.convBnVar_d,
			epsilon));
		return;
	}

	/*-----------Pooling-----------*/

	//void poolForward(int n, int c, int h, int w,
	//	value_type* srcData, value_type** dstData)
	//{
	//	/*const int poolDims = 2;
	//	int windowDimA[poolDims] = { 2,2 };
	//	int paddingA[poolDims] = { 0,0 };
	//	int strideA[poolDims] = { 2,2 };
	//	checkCUDNN(cudnnSetPoolingNdDescriptor(poolingDesc,
	//	CUDNN_POOLING_MAX,
	//	CUDNN_PROPAGATE_NAN,
	//	poolDims,
	//	windowDimA,
	//	paddingA,
	//	strideA));
	//	setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
	//	const int tensorDims = 4;
	//	int tensorOuputDimA[tensorDims] = { n,c,h,w };
	//	checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolingDesc,
	//	srcTensorDesc,
	//	tensorDims,
	//	tensorOuputDimA));
	//	n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
	//	h = tensorOuputDimA[2]; w = tensorOuputDimA[3];
	//	setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);*/
	//	resize(n*c*h*w, dstData);
	//	scaling_type alpha = scaling_type(1);
	//	scaling_type beta = scaling_type(0);
	//	checkCUDNN(cudnnPoolingForward(cudnnHandle,
	//		poolingDesc,
	//		&alpha,
	//		srcPoolTensorDesc,
	//		srcData,
	//		&beta,
	//		dstPoolTensorDesc,
	//		*dstData));
	//	//float* tmpDsc = new float[7 * 7];
	//	//memset(tmpDsc, 0, sizeof(float) * 7 * 7);
	//	//checkCudaErrors(cudaMemcpy(tmpDsc, *dstData, sizeof(float) * 7 * 7, cudaMemcpyDeviceToHost));
	//	//for (int j = 0; j < h; j++)
	//	//{
	//	//	for (int i = 0; i <w; i++)
	//	//	{
	//	//		//printf("%f ", tmpDsc[w*j +i]);
	//	//		cout << tmpDsc[w*j + i] << "\t";
	//	//	}
	//	//	printf("\n");
	//	//}
	//}

	/*----------softmax-----------*/

	//void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
	//{
	//	resize(n*c*h*w, dstData);
	//	/*setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
	//	setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);*/
	//	scaling_type alpha = scaling_type(1);
	//	scaling_type beta = scaling_type(0);
	//	checkCUDNN(cudnnSoftmaxForward(cudnnHandle,
	//		CUDNN_SOFTMAX_ACCURATE,
	//		CUDNN_SOFTMAX_MODE_CHANNEL,
	//		&alpha,
	//		srcSoftmaxTensorDesc,
	//		srcData,
	//		&beta,
	//		dstSoftmaxTensorDesc,
	//		*dstData));
	//	//float* tmpDsc = new float[10 * 1];
	//	//memset(tmpDsc, 0, sizeof(float) * 10 * 1);
	//	//checkCudaErrors(cudaMemcpy(tmpDsc, *dstData, sizeof(float) * 10 * 1, cudaMemcpyDeviceToHost));
	//	//for (int j = 0; j < 1; j++)
	//	//{
	//	//	for (int i = 0; i <10; i++)
	//	//	{
	//	//		//printf("%f ", tmpDsc[w*j +i]);
	//	//		cout << tmpDsc[w*j + i] << "\t";
	//	//	}
	//	//	printf("\n");
	//	//}
	//}

	void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
	{
		resize(n*c*h*w, dstData);
		setTensorDesc(activTensorDesc, tensorFormat, dataType, n, c, h, w);

		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		checkCUDNN(cudnnActivationForward(cudnnHandle,
			activDesc,
			&alpha,
			activTensorDesc,
			srcData,
			&beta,
			activTensorDesc,
			*dstData));
		return;
	}

	void activationForward_box(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
	{
		resize(n*c*h*w, dstData);
		setTensorDesc(activTensorDesc, tensorFormat, dataType, n, c, h, w);

		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		checkCUDNN(cudnnActivationForward(cudnnHandle,
			activBoxDesc,
			&alpha,
			activTensorDesc,
			srcData,
			&beta,
			activTensorDesc,
			*dstData));
		return;
	}

	void depthwise_conv(int *n, int *c, int *h, int *w, int padding, int stride, const int depthwiseOutputW, const Layer_t<value_type>& conv, value_type* srcData, value_type** dstData, value_type** tmp, const int conv_blocks_num,
		dim3 threads_per_block, dim3 num_of_blocks)
	{
		int depthFilterW = 3;
		const int padding_along_hw = (int)fmaxf((float)((depthwiseOutputW - 1)*stride + depthFilterW - *h), 0);
		const int padding_topLeft = (int)(padding_along_hw / 2);
		const int padding_bottomRight = padding_along_hw - padding_topLeft;
		const int padding_blocks = *h*(*c) / 32 + 1;
		int padded_h = *h + padding_along_hw;
		int padded_w = *w + padding_along_hw;

		dim3 padding_threads_num(32, 32, 1);
		dim3 padding_blocks_num(padding_blocks, padding_blocks, 1);

		depthSetting(1, conv.inputs, depthwiseOutputW, depthwiseOutputW, *h, padding_along_hw, dstData, tmp);
		padding_conv(srcData, *tmp, depthwiseOutputW, *h, *c, padding_topLeft, padding_bottomRight, padding_threads_num, padding_blocks_num);
		depth_f(srcData, *tmp, *dstData, conv.depthData_d, padded_h, padded_w, *c, depthFilterW, depthwiseOutputW, padding_along_hw, stride, conv_blocks_num, threads_per_block, num_of_blocks);
		return;
	}

	void pointwise_conv(int *n, int *c, int *h, int *w, int padding, int stride, const int depthwiseOutputW, const Layer_t<value_type>& conv, value_type*srcData, value_type** dstData, const int conv_blocks_num,
		dim3 threads_per_block, dim3 num_of_blocks) 
	{
		int depthFilterW = 3;
		pointSetting(1, conv.outputs, depthwiseOutputW, depthwiseOutputW, dstData);//&srcData
		point_f(srcData, *dstData, conv.pointData_d, *h, *c, depthFilterW, depthwiseOutputW, conv.outputs, padding, stride, conv_blocks_num, threads_per_block, num_of_blocks); //srcData
		checkCudaErrors(cudaDeviceSynchronize());
		return;
	}


	// ---------------------------------------------------------------------------------------------------//
	// -------------------------  Depthwise Separable Convolution Layer  ------------------------- //
	// ---------------------------------------------------------------------------------------------------//
	// *n : batches of input image
	// *c : channel of input image
	// *h , * w : height and width of input image
	// padding : Justify the padding(left&top and right&bottom) in depthwise_conv function.
	// conv : Layer 
	void dsc(int *n, int *c, int *h, int *w, int padding, int stride, const Layer_t<value_type>& conv, value_type** srcData, value_type** dstData, value_type** tmp)
	{
		int depthFilterW = 3;
		const int depthwiseOutputW = (int)((*h - depthFilterW + 2*padding) / stride + 1); //width and height of depthwise output(feature map).
		const int conv_blocks_num = (depthwiseOutputW) / 30 + 1;

		dim3 threads_per_block(30, 30, 1); //depthwise&pointwise thread setting
		dim3 num_of_blocks_depthwise(conv_blocks_num* *c, conv_blocks_num, 1); //Depthwise_conv block setting
		dim3 num_of_blocks_pointwise(conv_blocks_num* conv.outputs, conv_blocks_num, 1); //Pointwise_conv block setting

		/*---------------------Depthwise Convolution---------------------*/
		depthwise_conv(n, c, h, w, padding, stride, depthwiseOutputW, conv, *srcData, dstData, tmp, conv_blocks_num, threads_per_block, num_of_blocks_depthwise);
		resize(conv.inputs * depthwiseOutputW *depthwiseOutputW, srcData);
		batchNormalization(*dstData, *srcData, conv.depthBnBias_d, conv.depthBnScale_d, conv.depthBnMean_d, conv.depthBnVar_d, (float)epsilon, *c, depthwiseOutputW, depthwiseOutputW);
		//batchNorm_depth(1, conv.inputs, depthwiseOutputW, depthwiseOutputW, conv, srcTensorDesc, *dstData, srcData);
		activationForward(1, conv.inputs, depthwiseOutputW, depthwiseOutputW, *srcData, dstData);

		/*---------------------Pointwise Convolution---------------------*/
		pointwise_conv(n, c, h, w, padding, stride, depthwiseOutputW, conv, *dstData, srcData, conv_blocks_num, threads_per_block, num_of_blocks_pointwise);
		resize(conv.outputs * depthwiseOutputW *depthwiseOutputW, dstData);
		batchNormalization(*srcData, *dstData, conv.pointBnBias_d, conv.pointBnScale_d, conv.pointBnMean_d, conv.pointBnVar_d, (float)epsilon, conv.outputs, depthwiseOutputW, depthwiseOutputW);
		//batchNorm_point(1, conv.outputs, depthwiseOutputW, depthwiseOutputW, conv, dstTensorDesc, *srcData, dstData);
		activationForward(1, conv.outputs, depthwiseOutputW, depthwiseOutputW, *dstData, srcData);

		*n = 1; *c = conv.outputs;
		*h = *w = depthwiseOutputW;
		//cout << "dsc - n, c, h, w : " << *n <<"\t" << *c << "\t" << *h << "\t" << *w << endl;
		return;
	}

	void extraLayer(int *n, int *c, int *h, int *w, int padding, int stride, int *tensorOuputDimA, const int tensorDims, const Layer_t<value_type>& conv, value_type** srcData, value_type** dstData, value_type** tmp)
	{
		int convFilterW = 3;
		const int pointwiseOutputW = *h;
		const int convOutputW = (int)((*h - convFilterW + 2 * padding) / stride + 1);
		const int conv_blocks_num = (convOutputW) / 30 + 1;
		
		dim3 threads_per_block(30, 30, 1);
		dim3 num_of_blocks_pointwise(conv_blocks_num* conv.pointoutputs, conv_blocks_num, 1);

		pointSetting(1, conv.pointoutputs, pointwiseOutputW, pointwiseOutputW, dstData);
		point_f(*srcData, *dstData, conv.pointData_d, *h, *c, convFilterW, pointwiseOutputW, conv.pointoutputs, padding, stride, conv_blocks_num, threads_per_block, num_of_blocks_pointwise);
		
		//batchNorm_point(1, conv.pointoutputs, pointwiseOutputW, pointwiseOutputW, conv, srcTensorDesc, *dstData, srcData);
		batchNormalization(*dstData, *srcData, conv.pointBnBias_d, conv.pointBnScale_d, conv.pointBnMean_d, conv.pointBnVar_d, (float)epsilon, conv.pointoutputs, pointwiseOutputW, pointwiseOutputW);

		activationForward(1, conv.pointoutputs, pointwiseOutputW, pointwiseOutputW, *srcData, dstData);
		
		*n = 1; *c = conv.pointoutputs;
		*h = *w = pointwiseOutputW;

		convolutionSetting(n, c, h, w, conv, padding, stride, tensorOuputDimA, tensorDims, srcData);

		resize(*c * *h * *w, srcData);
		float * filter = conv.convData_d;
		convolution(dstData, tmp, srcData, &filter, pointwiseOutputW, pointwiseOutputW, conv.pointoutputs, conv.outputs, 3, padding, stride);

		//batchNorm_conv(*n, *c, *h, *w, conv, dstTensorDesc, *srcData, dstData);
		resize(1**c**h**w, dstData);
		batchNormalization(*srcData, *dstData, conv.convBnBias_d, conv.convBnScale_d, conv.convBnMean_d, conv.convBnVar_d, (float)epsilon, conv.outputs, *h, *w);
		
		activationForward(*n, *c, *h, *w, *dstData, srcData);
		return;
	}

	void boxPredictor(int *n, int *c, int *h, int *w, int original_image_h, int original_image_w, int *anchor_num, int *box_featuremap_size, int* box_index, int* count_layer, const Layer_t<value_type>&conv_loc,
		const Layer_t<value_type>&conv_conf, value_type** anchorShape, value_type** srcData, value_type** locData, value_type** confData, value_type** locData_all, value_type** confData_all)
	{
		const int conv_blocks_num = (*h) / 30 + 1;
		dim3 threads_per_block(30, 30, 1);
		dim3 num_of_blocks_conf(conv_blocks_num * conv_conf.outputs, conv_blocks_num, 1);
		dim3 num_of_blocks_loc(conv_blocks_num * conv_loc.outputs, conv_blocks_num, 1);
		value_type *confData_tmp = NULL;
		
		pointSetting(1, conv_loc.outputs, *h, *w, locData);
		pointbias_f(*srcData, *locData, conv_loc.pointData_d, conv_loc.pointBias_d, *h, *c, 1, *h, conv_loc.outputs, 0, 0, conv_blocks_num, threads_per_block, num_of_blocks_loc);

		pointSetting(1, conv_conf.outputs, *h, *w, &confData_tmp);
		pointbias_f(*srcData, confData_tmp, conv_conf.pointData_d, conv_conf.pointBias_d, *h, *c, 1, *h, conv_conf.outputs, 0, 0, conv_blocks_num, threads_per_block, num_of_blocks_conf);

		remove_background(confData_tmp, anchor_num[*box_index], num_classes, *w, threads_per_block, num_of_blocks_conf);
		activationForward_box(*n, conv_conf.outputs, *h, *w, confData_tmp, confData);

		encode_boxData(*h, *w, original_image_h, original_image_w, anchor_num[*box_index], count_layer, anchorShape, locData);

		clipWindow(locData, anchor_num[*box_index], *h, *w, original_image_h, original_image_w);

		sum_boxes(*locData, *confData, *locData_all, *confData_all, num_classes, box_featuremap_size, anchor_num, *box_index, box_code, box_total);

		*box_index += 1;
		checkCudaErrors(cudaFree(confData_tmp));
		return;
	}

	void encode_boxData(int h, int w, int original_image_h, int original_image_w, int num_anchors, int* count_layer, value_type** anchorShape, value_type ** locData)
	{
		int channels = box_code * num_anchors;
		dim3 threads_per_block(w, h, 1);
		dim3 num_of_blocks(1, channels, 1);
		
		encode_locData(*locData, num_anchors, *anchorShape, box_code, h, w, original_image_h, original_image_w, *count_layer, threads_per_block, num_of_blocks);

		*count_layer += 1;
		return;
	}

	void clipWindow(float **locData, int num_anchors, int featuremap_height, int featuremap_width, int original_image_h, int original_image_w)
	{
		dim3 threads_per_block(featuremap_width, featuremap_height, 1);
		dim3 num_of_blocks(1, num_anchors, 1);
		clip_window(*locData, num_anchors, box_code, featuremap_height, featuremap_width, original_image_h, original_image_w, threads_per_block, num_of_blocks);
		return;
	}

	int classify_example(Mat* input_image, Mat* output_image, string* result_word, const Layer_t<value_type>& conv0, const Layer_t<value_type>& conv1, const Layer_t<value_type>& conv2, const Layer_t<value_type>& conv3,
		const Layer_t<value_type>& conv4, const Layer_t<value_type>& conv5, const Layer_t<value_type>& conv6, const Layer_t<value_type>& conv7, const Layer_t<value_type>& conv8,
		const Layer_t<value_type>& conv9, const Layer_t<value_type>& conv10, const Layer_t<value_type>& conv11, const Layer_t<value_type>& conv12, const Layer_t<value_type>& conv13,
		const Layer_t<value_type>& conv14, const Layer_t<value_type>& conv15, const Layer_t<value_type>& conv16, const Layer_t<value_type>& conv17, const Layer_t<value_type>& box0_loc,
		const Layer_t<value_type>& box0_conf, const Layer_t<value_type>& box1_loc, const Layer_t<value_type>& box1_conf, const Layer_t<value_type>& box2_loc, const Layer_t<value_type>& box2_conf,
		const Layer_t<value_type>& box3_loc, const Layer_t<value_type>& box3_conf, const Layer_t<value_type>& box4_loc, const Layer_t<value_type>& box4_conf,
		const Layer_t<value_type>& box5_loc, const Layer_t<value_type>& box5_conf) //const char* fname
	{
		clock_t start, end;
		double single_start, single_end;
		int n, c, h, w;
		int layer_count = 0;
		int box_index = 0;
		int original_image_h = NULL;
		int original_image_w = NULL;
		int box0_h, box0_w, box1_h, box1_w, box2_h, box2_w, box3_h, box3_w, box4_h, box4_w, box5_h, box5_w;
		value_type *srcData = NULL;
		value_type *dstData = NULL;
		value_type *tmp = NULL;
		value_type *conv11_locData = NULL; value_type *conv11_confData = NULL; value_type *conv13_locData = NULL; value_type *conv13_confData = NULL;
		value_type *conv14_locData = NULL; value_type *conv14_confData = NULL; value_type *conv15_locData = NULL; value_type *conv15_confData = NULL;
		value_type *conv16_locData = NULL; value_type *conv16_confData = NULL; value_type *conv17_locData = NULL; value_type *conv17_confData = NULL;
		value_type *confData_all = NULL; value_type *locData_all = NULL;

		resize(box_total * (num_classes + 1), &confData_all);
		resize(box_total * box_code, &locData_all);
		checkCudaErrors(cudaMalloc(&srcData, sizeof(value_type)*IMAGE_H*IMAGE_W * IMAGE_C));

		int *box_featuremap_size = new int[6];
		box_featuremap_size[0] = 19;
		box_featuremap_size[1] = 10;
		box_featuremap_size[2] = 5;
		box_featuremap_size[3] = 3;
		box_featuremap_size[4] = 2;
		box_featuremap_size[5] = 1;

		int *anchor_num = new int[6];
		anchor_num[0] = 3;
		for (int i = 1; i < 6; i++) {
			anchor_num[i] = 6;
		}

		

		//std::cout << "Performing forward propagation ...\n";

		n = 1; c = IMAGE_C; h = IMAGE_H; w = IMAGE_W;
		const int tensorDims = 4;
		int tensorOuputDimA[tensorDims];
		//int num_anchors = 3;

		////-----activation pre-processing-----////
		checkCUDNN(cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, 6.0));
		checkCUDNN(cudnnSetActivationDescriptor(activBoxDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));

		//////-----pooling pre-processing-----////

		//const int poolDims = 2;
		//int windowDimA[poolDims] = { 2,2 };
		//int paddingA[poolDims] = { 0,0 };
		//int strideA[poolDims] = { 2,2 };
		//checkCUDNN(cudnnSetPoolingNdDescriptor(poolingDesc,
		//	CUDNN_POOLING_MAX,
		//	CUDNN_PROPAGATE_NAN,
		//	poolDims,
		//	windowDimA,
		//	paddingA,
		//	strideA));
		//setTensorDesc(srcPoolTensorDesc, tensorFormat, dataType, n, c, h, w);
		////const int tensorDims = 4;
		////int tensorOuputDimA[tensorDims] = { n,c,h,w };
		//checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolingDesc,
		//	srcPoolTensorDesc,
		//	tensorDims,
		//	tensorOuputDimA));
		//n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
		//h = tensorOuputDimA[2]; w = tensorOuputDimA[3];
		//setTensorDesc(dstPoolTensorDesc, tensorFormat, dataType, n, c, h, w);

		//////----softmax pre-processing-----////

		//setTensorDesc(srcSoftmaxTensorDesc, tensorFormat, dataType, 1, 10, 1, 1);
		//setTensorDesc(dstSoftmaxTensorDesc, tensorFormat, dataType, 1, 10, 1, 1);

		value_type* aspect_ratio = new value_type[num_of_featuremaps];
		value_type* aspect_ratio_layer0 = new value_type[(const int)(num_of_featuremaps / 2)];
		value_type *anchorShape0 = NULL; //device memory
		value_type *anchorShape1to5 = NULL; //device memory

		anchorBox_generator(aspect_ratio_layer0, aspect_ratio, &anchorShape0, &anchorShape1to5);

		convolutionSetting(&n, &c, &h, &w, conv0, 1, 2, tensorOuputDimA, tensorDims, &dstData);

		start = clock();
		image_resize(input_image, &srcData, &original_image_h, &original_image_w, IMAGE_H, IMAGE_W); //fname

		float * filter = conv0.convData_d;
		convolution(&srcData, &tmp, &dstData, &filter, 300, 300, 3, 32, 3, 1, 2);

		resize(c * h * w, &srcData);
		batchNormalization(dstData, srcData, conv0.convBnBias_d, conv0.convBnScale_d, conv0.convBnMean_d, conv0.convBnVar_d, (float)epsilon, conv0.outputs, h, w);
		//batchNorm_conv(n, c, h, w, conv0, srcTensorDesc, dstData, &srcData);
		
		resize(c * h *w, &dstData);
		activationForward(n, c, h, w, srcData, &dstData);
		
		dsc(&n, &c, &h, &w, 1, 1, conv1, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 2, conv2, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 1, conv3, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 2, conv4, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 1, conv5, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 2, conv6, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 1, conv7, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 1, conv8, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 1, conv9, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 1, conv10, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 1, conv11, &dstData, &srcData, &tmp);

		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box0_loc, box0_conf, &anchorShape0,
			&dstData, &conv11_locData, &conv11_confData, &locData_all, &confData_all);

		layer_count = 0;

		dsc(&n, &c, &h, &w, 1, 2, conv12, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 1, conv13, &dstData, &srcData, &tmp);

		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box1_loc, box1_conf, &anchorShape1to5,
			&dstData, &conv13_locData, &conv13_confData, &locData_all, &confData_all);

		extraLayer(&n, &c, &h, &w, 1, 2, tensorOuputDimA, tensorDims, conv14, &dstData, &srcData, &tmp);
		
		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box2_loc, box2_conf, &anchorShape1to5,
			&dstData, &conv14_locData, &conv14_confData, &locData_all, &confData_all);

		extraLayer(&n, &c, &h, &w, 1, 2, tensorOuputDimA, tensorDims, conv15, &dstData, &srcData, &tmp);
		
		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box3_loc, box3_conf, &anchorShape1to5,
			&dstData, &conv15_locData, &conv15_confData, &locData_all, &confData_all);

		extraLayer(&n, &c, &h, &w, 1, 2, tensorOuputDimA, tensorDims, conv16, &dstData, &srcData, &tmp);
		
		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box4_loc, box4_conf, &anchorShape1to5,
			&dstData, &conv16_locData, &conv16_confData, &locData_all, &confData_all);

		extraLayer(&n, &c, &h, &w, 1, 2, tensorOuputDimA, tensorDims, conv17, &dstData, &srcData, &tmp);

		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box5_loc, box5_conf, &anchorShape1to5,
			&dstData, &conv17_locData, &conv17_confData, &locData_all, &confData_all);

		float* tmDsc12 = new float[box_total * (num_classes+1)];
		memset(tmDsc12, 0, sizeof(float)* box_total * (num_classes + 1));
		cudaMemcpy(tmDsc12, confData_all, sizeof(int)* box_total * (num_classes + 1), cudaMemcpyDeviceToHost);
		float* tmDsc13 = new float[box_total * box_code];
		memset(tmDsc13, 0, sizeof(float) * box_total * box_code);
		cudaMemcpy(tmDsc13, locData_all, sizeof(float) * box_total * box_code, cudaMemcpyDeviceToHost);
		int* sorted_box_address = new int[box_total];
		memset(sorted_box_address, 0, sizeof(int)* box_total);
		int* sorted_box_class = new int[box_total];
		memset(sorted_box_class, 0, sizeof(int)* box_total);
		
		
		int sorted_box_index = 0;
		
		for (int i = 0; i < num_classes+1; ++i) {
			for (int j = 0; j < box_total; ++j) {
				if (tmDsc12[i * box_total + j] > score_threshold) {
					sorted_box_address[sorted_box_index] = j;
					sorted_box_class[sorted_box_index] = i;
					sorted_box_index += 1;
				}
			}
		}
		float* filtered_box_score = new float[sorted_box_index];
		memset(filtered_box_score, 0, sizeof(float)* sorted_box_index);
		float* sorted_box_score = new float[sorted_box_index];
		memset(sorted_box_score, 0, sizeof(float)* sorted_box_index);
		int* sorted_score_index = new int[sorted_box_index];
		memset(sorted_score_index, 0, sizeof(int)* sorted_box_index);

		for (int i = 0; i < sorted_box_index; i++) {
			sorted_box_score[i] = tmDsc12[sorted_box_class[i] * box_total + sorted_box_address[i]];
			filtered_box_score[i] = sorted_box_score[i];
		}

		qsort(sorted_box_score, sorted_box_index, sizeof(float), descending);

		for (int i = 0; i < sorted_box_index; i++) {
			sorted_score_index[i] = find_index(sorted_box_score, sorted_box_index, filtered_box_score[i]);
			//cout << sorted_box_address[sorted_score_index[i]] << endl;
		}
		
		vector<int> address_index(sorted_box_index);
		vector<float> score_index(sorted_box_index);
		vector<pair <int, int>> address_class(sorted_box_index);
		for (int i = 0; i < sorted_box_index; i++){
			score_index[i] = sorted_box_score[i];
			//cout << sorted_box_class[sorted_score_index[i]] << endl;
			address_index[i] = sorted_box_address[sorted_score_index[i]];
			address_class[i] = make_pair(address_index[i], sorted_box_class[sorted_score_index[i]]);
		}

		if (address_class.size() == 0) {

		}
		else {
			nms(tmDsc13, address_class, 0.6, box_total);

			draw_allboxes(output_image, tmDsc13, address_class, box_total);

			vector<int> sequence;
			/*-----------result_word------------*/
			sequence = sort_by_sequence(result_word, tmDsc13, address_class, box_total);
			
			for (int i = 0; i < address_class.size(); i++) {
				*result_word += string(class_name[sequence[i]]);
			}
			vector<int>().swap(sequence);
		}
		

		end = clock();
		std::cout << "\nSingle computing time : " << (end - start) << std::endl;
		
		/*-------free memory-------*/
		vector<int>().swap(address_index);
		vector<float>().swap(score_index);
		vector<pair <int,int>>().swap(address_class);
		delete[] tmDsc12;
		delete[] tmDsc13;
		delete[] sorted_box_address;
		delete[] sorted_box_class;
		delete[] filtered_box_score;
		delete[] sorted_box_score;
		delete[] sorted_score_index;
		delete[] aspect_ratio_layer0;
		delete[] aspect_ratio;
		delete[] anchor_num;
		delete[] box_featuremap_size;
		checkCudaErrors(cudaFree(srcData));
		checkCudaErrors(cudaFree(dstData));
		checkCudaErrors(cudaFree(tmp));
		checkCudaErrors(cudaFree(conv11_locData));
		checkCudaErrors(cudaFree(conv11_confData));
		checkCudaErrors(cudaFree(conv13_locData));
		checkCudaErrors(cudaFree(conv13_confData));
		checkCudaErrors(cudaFree(conv14_locData));
		checkCudaErrors(cudaFree(conv14_confData));
		checkCudaErrors(cudaFree(conv15_locData));
		checkCudaErrors(cudaFree(conv15_confData));
		checkCudaErrors(cudaFree(conv16_locData));
		checkCudaErrors(cudaFree(conv16_confData));
		checkCudaErrors(cudaFree(conv17_locData));
		checkCudaErrors(cudaFree(conv17_confData));
		checkCudaErrors(cudaFree(confData_all));
		checkCudaErrors(cudaFree(locData_all));
		checkCudaErrors(cudaFree(anchorShape0));
		checkCudaErrors(cudaFree(anchorShape1to5));

		return 0;
	}
};

#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
// using 1x1 convolution to emulate gemv in half precision when cuBLAS version <= 7.0
template <>
void network_t<half1>::fullyConnectedForward(const Layer_t<half1>& ip,
	int& n, int& c, int& h, int& w,
	half1* srcData, half1** dstData)
{
	c = c*h*w; h = 1; w = 1;
	network_t<half1>::convoluteForward(ip, n, c, h, w, srcData, dstData);
	c = ip.outputs;
}
#endif


//
//int main(int argc, char *argv[])
//{
//	std::string image_path;
//	clock_t start, end;
//	float single_start, single_end, half_start, half_end;
//
//
//	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
//	{
//		displayUsage();
//		exit(EXIT_WAIVED);
//	}
//
//	int version = (int)cudnnGetVersion();
//	printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
//	printf("Host compiler version : %s %s\r", COMPILER_NAME, COMPILER_VER);
//	showDevices();
//
//	int device = 0;
//	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
//	{
//		device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
//		checkCudaErrors(cudaSetDevice(device));
//	}
//	std::cout << "Using device " << device << std::endl;
//
//	if (checkCmdLineFlag(argc, (const char **)argv, "image"))
//	{
//		
//		char* image_name;
//		getCmdLineArgumentString(argc, (const char **)argv, "image", (char **)&image_name);
//
//		network_t<float> ssd_mobilenet;
//		//convi(input, output, depth_dim, point_dim)
//		Layer_t<float> conv0(3, 32, 3, conv0_bin, bnScale0, bnBias0, conv0_mMean, conv0_mVar, argv[0]);
//		Layer_t<float> conv1(32, 64, 3, 1, conv1_depth, conv1_point, conv1_dGamma, conv1_dBeta, conv1_pGamma, conv1_pBeta, conv1_dmMean, conv1_dmVar, conv1_pmMean, conv1_pmVar, argv[0]);
//		Layer_t<float> conv2(64, 128, 3, 1, conv2_depth, conv2_point, conv2_dGamma, conv2_dBeta, conv2_pGamma, conv2_pBeta, conv2_dmMean, conv2_dmVar, conv2_pmMean, conv2_pmVar, argv[0]);
//		Layer_t<float> conv3(128, 128, 3, 1, conv3_depth, conv3_point, conv3_dGamma, conv3_dBeta, conv3_pGamma, conv3_pBeta, conv3_dmMean, conv3_dmVar, conv3_pmMean, conv3_pmVar, argv[0]);
//		Layer_t<float> conv4(128, 256, 3, 1, conv4_depth, conv4_point, conv4_dGamma, conv4_dBeta, conv4_pGamma, conv4_pBeta, conv4_dmMean, conv4_dmVar, conv4_pmMean, conv4_pmVar, argv[0]);
//		Layer_t<float> conv5(256, 256, 3, 1, conv5_depth, conv5_point, conv5_dGamma, conv5_dBeta, conv5_pGamma, conv5_pBeta, conv5_dmMean, conv5_dmVar, conv5_pmMean, conv5_pmVar, argv[0]);
//		Layer_t<float> conv6(256, 512, 3, 1, conv6_depth, conv6_point, conv6_dGamma, conv6_dBeta, conv6_pGamma, conv6_pBeta, conv6_dmMean, conv6_dmVar, conv6_pmMean, conv6_pmVar, argv[0]);
//		Layer_t<float> conv7(512, 512, 3, 1, conv7_depth, conv7_point, conv7_dGamma, conv7_dBeta, conv7_pGamma, conv7_pBeta, conv7_dmMean, conv7_dmVar, conv7_pmMean, conv7_pmVar, argv[0]);
//		Layer_t<float> conv8(512, 512, 3, 1, conv8_depth, conv8_point, conv8_dGamma, conv8_dBeta, conv8_pGamma, conv8_pBeta, conv8_dmMean, conv8_dmVar, conv8_pmMean, conv8_pmVar, argv[0]);
//		Layer_t<float> conv9(512, 512, 3, 1, conv9_depth, conv9_point, conv9_dGamma, conv9_dBeta, conv9_pGamma, conv9_pBeta, conv9_dmMean, conv9_dmVar, conv9_pmMean, conv9_pmVar, argv[0]);
//		Layer_t<float> conv10(512, 512, 3, 1, conv10_depth, conv10_point, conv10_dGamma, conv10_dBeta, conv10_pGamma, conv10_pBeta, conv10_dmMean, conv10_dmVar, conv10_pmMean, conv10_pmVar, argv[0]);
//		Layer_t<float> conv11(512, 512, 3, 1, conv11_depth, conv11_point, conv11_dGamma, conv11_dBeta, conv11_pGamma, conv11_pBeta, conv11_dmMean, conv11_dmVar, conv11_pmMean, conv11_pmVar, argv[0]);
//		Layer_t<float> conv12(512,1024, 3, 1, conv12_depth, conv12_point, conv12_dGamma, conv12_dBeta, conv12_pGamma, conv12_pBeta, conv12_dmMean, conv12_dmVar, conv12_pmMean, conv12_pmVar, argv[0]);
//		Layer_t<float> conv13(1024, 1024, 3, 1, conv13_depth, conv13_point, conv13_dGamma, conv13_dBeta, conv13_pGamma, conv13_pBeta, conv13_dmMean, conv13_dmVar, conv13_pmMean, conv13_pmVar, argv[0]);
//		Layer_t<float> conv14(1024, 256, 512, 3, 1, conv14_w, conv14_point, conv14_wGamma, conv14_wBeta, conv14_pGamma, conv14_pBeta, conv14_wmMean, conv14_wmVar, conv14_pmMean, conv14_pmVar, argv[0]);
//		Layer_t<float> conv15(512, 128, 256, 3, 1, conv15_w, conv15_point, conv15_wGamma, conv15_wBeta, conv15_pGamma, conv15_pBeta, conv15_wmMean, conv15_wmVar, conv15_pmMean, conv15_pmVar, argv[0]);
//		Layer_t<float> conv16(256, 128, 256, 3, 1, conv16_w, conv16_point, conv16_wGamma, conv16_wBeta, conv16_pGamma, conv16_pBeta, conv16_wmMean, conv16_wmVar, conv16_pmMean, conv16_pmVar, argv[0]);
//		Layer_t<float> conv17(256, 64, 128, 3, 1, conv17_w, conv17_point, conv17_wGamma, conv17_wBeta, conv17_pGamma, conv17_pBeta, conv17_wmMean, conv17_wmVar, conv17_pmMean, conv17_pmVar, argv[0]);
//		Layer_t<float> box0_loc(512, 12, 1, box0_loc_w, box0_loc_b, argv[0]);
//		Layer_t<float> box0_conf(512, (num_classes + 1) * 3, 1, box0_conf_w, box0_conf_b, argv[0]);
//		Layer_t<float> box1_loc(1024, 24, 1, box1_loc_w, box1_loc_b, argv[0]);
//		Layer_t<float> box1_conf(1024, (num_classes + 1) * 6, 1, box1_conf_w, box1_conf_b, argv[0]);
//		Layer_t<float> box2_loc(512, 24, 1, box2_loc_w, box2_loc_b, argv[0]);
//		Layer_t<float> box2_conf(512, (num_classes + 1) * 6, 1, box2_conf_w, box2_conf_b, argv[0]);
//		Layer_t<float> box3_loc(256, 24, 1, box3_loc_w, box3_loc_b, argv[0]);
//		Layer_t<float> box3_conf(256, (num_classes + 1) * 6, 1, box3_conf_w, box3_conf_b, argv[0]);
//		Layer_t<float> box4_loc(256, 24, 1, box4_loc_w, box4_loc_b, argv[0]);
//		Layer_t<float> box4_conf(256, (num_classes + 1) * 6, 1, box4_conf_w, box4_conf_b, argv[0]);
//		Layer_t<float> box5_loc(128, 24, 1, box5_loc_w, box5_loc_b, argv[0]);
//		Layer_t<float> box5_conf(128, (num_classes + 1) * 6, 1, box5_conf_w, box5_conf_b, argv[0]);
//		
//		ssd_mobilenet.classify_example(image_name, conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11,
//			conv12, conv13, conv14, conv15, conv16, conv17, box0_loc, box0_conf, box1_loc, box1_conf, box2_loc, box2_conf, box3_loc, box3_conf, box4_loc, box4_conf, box5_loc, box5_conf); 		
//
//		cudaDeviceReset();
//
//		exit(EXIT_SUCCESS);
//	}
//
//	// default behaviour
//	if (argc == 1 || (argc == 2) && checkCmdLineFlag(argc, (const char **)argv, "device"))
//	{
//		// check available memory
//		struct cudaDeviceProp prop;
//		checkCudaErrors(cudaGetDeviceProperties(&prop, device));
//		double globalMem = prop.totalGlobalMem / double(1024 * 1024);
//		bool low_memory = false;
//		if (globalMem < 1536)
//		{
//			// takes care of 1x1 convolution workaround for fully connected layers
//			// when CUDNN_CONVOLUTION_FWD_ALGO_FFT is used
//#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
//			low_memory = true;
//#endif
//		}
//		{
//			
//			std::cout << "\nTesting single precision\n";
//			network_t<float> ssd_mobilenet;
//			Layer_t<float> conv0(3, 32, 3, conv0_bin, bnScale0, bnBias0, conv0_mMean, conv0_mVar, argv[0]); 
//			Layer_t<float> conv1(32, 64, 3, 1, conv1_depth, conv1_point, conv1_dGamma, conv1_dBeta, conv1_pGamma, conv1_pBeta, conv1_dmMean, conv1_dmVar, conv1_pmMean, conv1_pmVar, argv[0]);
//			Layer_t<float> conv2(64, 128, 3, 1, conv2_depth, conv2_point, conv2_dGamma, conv2_dBeta, conv2_pGamma, conv2_pBeta, conv2_dmMean, conv2_dmVar, conv2_pmMean, conv2_pmVar, argv[0]);
//			Layer_t<float> conv3(128, 128, 3, 1, conv3_depth, conv3_point, conv3_dGamma, conv3_dBeta, conv3_pGamma, conv3_pBeta, conv3_dmMean, conv3_dmVar, conv3_pmMean, conv3_pmVar, argv[0]);
//			Layer_t<float> conv4(128, 256, 3, 1, conv4_depth, conv4_point, conv4_dGamma, conv4_dBeta, conv4_pGamma, conv4_pBeta, conv4_dmMean, conv4_dmVar, conv4_pmMean, conv4_pmVar, argv[0]);
//			Layer_t<float> conv5(256, 256, 3, 1, conv5_depth, conv5_point, conv5_dGamma, conv5_dBeta, conv5_pGamma, conv5_pBeta, conv5_dmMean, conv5_dmVar, conv5_pmMean, conv5_pmVar, argv[0]);
//			Layer_t<float> conv6(256, 512, 3, 1, conv6_depth, conv6_point, conv6_dGamma, conv6_dBeta, conv6_pGamma, conv6_pBeta, conv6_dmMean, conv6_dmVar, conv6_pmMean, conv6_pmVar, argv[0]);
//			Layer_t<float> conv7(512, 512, 3, 1, conv7_depth, conv7_point, conv7_dGamma, conv7_dBeta, conv7_pGamma, conv7_pBeta, conv7_dmMean, conv7_dmVar, conv7_pmMean, conv7_pmVar, argv[0]);
//			Layer_t<float> conv8(512, 512, 3, 1, conv8_depth, conv8_point, conv8_dGamma, conv8_dBeta, conv8_pGamma, conv8_pBeta, conv8_dmMean, conv8_dmVar, conv8_pmMean, conv8_pmVar, argv[0]);
//			Layer_t<float> conv9(512, 512, 3, 1, conv9_depth, conv9_point, conv9_dGamma, conv9_dBeta, conv9_pGamma, conv9_pBeta, conv9_dmMean, conv9_dmVar, conv9_pmMean, conv9_pmVar, argv[0]);
//			Layer_t<float> conv10(512, 512, 3, 1, conv10_depth, conv10_point, conv10_dGamma, conv10_dBeta, conv10_pGamma, conv10_pBeta, conv10_dmMean, conv10_dmVar, conv10_pmMean, conv10_pmVar, argv[0]);
//			Layer_t<float> conv11(512, 512, 3, 1, conv11_depth, conv11_point, conv11_dGamma, conv11_dBeta, conv11_pGamma, conv11_pBeta, conv11_dmMean, conv11_dmVar, conv11_pmMean, conv11_pmVar, argv[0]);
//			Layer_t<float> conv12(512, 1024, 3, 1, conv12_depth, conv12_point, conv12_dGamma, conv12_dBeta, conv12_pGamma, conv12_pBeta, conv12_dmMean, conv12_dmVar, conv12_pmMean, conv12_pmVar, argv[0]);
//			Layer_t<float> conv13(1024, 1024, 3, 1, conv13_depth, conv13_point, conv13_dGamma, conv13_dBeta, conv13_pGamma, conv13_pBeta, conv13_dmMean, conv13_dmVar, conv13_pmMean, conv13_pmVar, argv[0]);
//			Layer_t<float> conv14(1024, 256, 512, 3, 1, conv14_w, conv14_point, conv14_wGamma, conv14_wBeta, conv14_pGamma, conv14_pBeta, conv14_wmMean, conv14_wmVar, conv14_pmMean, conv14_pmVar, argv[0]);
//			Layer_t<float> conv15(512, 128, 256, 3, 1, conv15_w, conv15_point, conv15_wGamma, conv15_wBeta, conv15_pGamma, conv15_pBeta, conv15_wmMean, conv15_wmVar, conv15_pmMean, conv15_pmVar, argv[0]);
//			Layer_t<float> conv16(256, 128, 256, 3, 1, conv16_w, conv16_point, conv16_wGamma, conv16_wBeta, conv16_pGamma, conv16_pBeta, conv16_wmMean, conv16_wmVar, conv16_pmMean, conv16_pmVar, argv[0]);
//			Layer_t<float> conv17(256, 64, 128, 3, 1, conv17_w, conv17_point, conv17_wGamma, conv17_wBeta, conv17_pGamma, conv17_pBeta, conv17_wmMean, conv17_wmVar, conv17_pmMean, conv17_pmVar, argv[0]);
//			Layer_t<float> box0_loc(512, 12, 1, box0_loc_w, box0_loc_b, argv[0]);
//			Layer_t<float> box0_conf(512, (num_classes + 1) * 3, 1, box0_conf_w, box0_conf_b, argv[0]);
//			Layer_t<float> box1_loc(1024, 24, 1, box1_loc_w, box1_loc_b, argv[0]);
//			Layer_t<float> box1_conf(1024, (num_classes + 1) * 6, 1, box1_conf_w, box1_conf_b, argv[0]);
//			Layer_t<float> box2_loc(512, 24, 1, box2_loc_w, box2_loc_b, argv[0]);
//			Layer_t<float> box2_conf(512, (num_classes + 1) * 6, 1, box2_conf_w, box2_conf_b, argv[0]);
//			Layer_t<float> box3_loc(256, 24, 1, box3_loc_w, box3_loc_b, argv[0]);
//			Layer_t<float> box3_conf(256, (num_classes + 1) * 6, 1, box3_conf_w, box3_conf_b, argv[0]);
//			Layer_t<float> box4_loc(256, 24, 1, box4_loc_w, box4_loc_b, argv[0]);
//			Layer_t<float> box4_conf(256, (num_classes + 1) * 6, 1, box4_conf_w, box4_conf_b, argv[0]);
//			Layer_t<float> box5_loc(128, 24, 1, box5_loc_w, box5_loc_b, argv[0]);
//			Layer_t<float> box5_conf(128, (num_classes + 1) * 6, 1, box5_conf_w, box5_conf_b, argv[0]);
//
//			get_path(image_path, input_image, argv[0]);
//			for (int i = 0; i < 100; i++)
//			{
//				ssd_mobilenet.classify_example(image_path.c_str(), conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
//					conv13, conv14, conv15, conv16, conv17, box0_loc, box0_conf, box1_loc, box1_conf, box2_loc, box2_conf, box3_loc, box3_conf, box4_loc, box4_conf, box5_loc, box5_conf);
//			}
//
//			std::cout << "\n Test finished" << std::endl;
//		}
//
//		//{
//		//	half_start = clock();
//		//	std::cout << "\nTesting half precision (math in single precision)\n";
//		//	network_t<half1> mnist;
//		//	// Conversion of input weights to half precision is done
//		//	// on host using tools from fp16_emu.cpp
//		//	Layer_t<half1> conv1(1, 1, 2, conv1_bin, conv1_bias_bin, argv[0], FP16_HOST);
//
//		//	// Conversion of input weights to half precision is done
//		//	// on device using cudnnTransformTensor
//		//	Layer_t<half1>   ip1(1, 10, 7, ip1_bin, ip1_bias_bin, argv[0], FP16_CUDNN);
//		//	// Conversion of input weights to half precision is done
//		//	// on device using CUDA kernel from fp16_dev.cu
//		//	get_path(image_path, first_image, argv[0]);
//		//	i1 = mnist.classify_example(image_path.c_str(), conv1, ip1);
//		//	get_path(image_path, input_image, argv[0]);
//		//	i2 = mnist.classify_example(image_path.c_str(), conv1, ip1);
//		//	get_path(image_path, third_image, argv[0]);
//		//	// New feature in cuDNN v3: FFT for convolution
//		//	i3 = mnist.classify_example(image_path.c_str(), conv1, ip1);
//		//	std::cout << "\nResult of classification: " << i1 << " " << i2 << " " << i3 << std::endl;
//		//	if (i1 != 1 || i2 != 3 || i3 != 5)
//		//	{
//		//		std::cout << "\nTest failed!\n";
//		//		system("pause");
//		//		FatalError("Prediction mismatch");
//		//		
//		//	}
//		//	else
//		//	{
//		//		half_end = clock();
//		//		std::cout << "\nHalf computing time: " << (half_end - half_start) / CLOCKS_PER_SEC << std::endl;
//		//		std::cout << "\nTest passed!\n";
//		//	}
//		//}
//		//cudaDeviceReset();
//
//		exit(EXIT_SUCCESS);
//	}
//
//	displayUsage();
//	cudaDeviceReset();
//
//	exit(EXIT_WAIVED);
//}
