#include "OCR.h"

using namespace cv;
using namespace std;

const char *conv0_bin = "binData/conv0_d.bin";
const char *bnScale0 = "binData/conv0_dgamma.bin";
const char *bnBias0 = "binData/conv0_dbeta.bin";
const char *conv0_mMean = "binData/conv0_dMean.bin";
const char *conv0_mVar = "binData/conv0_dVar.bin";

const char *conv1_depth = "binData/conv1_d.bin";
const char *conv1_point = "binData/conv1_p.bin";
const char *conv1_dBeta = "binData/conv1_dbeta.bin";
const char *conv1_pBeta = "binData/conv1_pbeta.bin";
const char *conv1_dGamma = "binData/conv1_dgamma.bin";
const char *conv1_pGamma = "binData/conv1_pgamma.bin";
const char *conv1_dmMean = "binData/conv1_dMean.bin";
const char *conv1_dmVar = "binData/conv1_dVar.bin";
const char *conv1_pmMean = "binData/conv1_pMean.bin";
const char *conv1_pmVar = "binData/conv1_pVar.bin";

const char *conv2_depth = "binData/conv2_d.bin";
const char *conv2_point = "binData/conv2_p.bin";
const char *conv2_dBeta = "binData/conv2_dbeta.bin";
const char *conv2_pBeta = "binData/conv2_pbeta.bin";
const char *conv2_dGamma = "binData/conv2_dgamma.bin";
const char *conv2_pGamma = "binData/conv2_pgamma.bin";
const char *conv2_dmMean = "binData/conv2_dMean.bin";
const char *conv2_dmVar = "binData/conv2_dVar.bin";
const char *conv2_pmMean = "binData/conv2_pMean.bin";
const char *conv2_pmVar = "binData/conv2_pVar.bin";

const char *conv3_depth = "binData/conv3_d.bin";
const char *conv3_point = "binData/conv3_p.bin";
const char *conv3_dBeta = "binData/conv3_dbeta.bin";
const char *conv3_pBeta = "binData/conv3_pbeta.bin";
const char *conv3_dGamma = "binData/conv3_dgamma.bin";
const char *conv3_pGamma = "binData/conv3_pgamma.bin";
const char *conv3_dmMean = "binData/conv3_dMean.bin";
const char *conv3_dmVar = "binData/conv3_dVar.bin";
const char *conv3_pmMean = "binData/conv3_pMean.bin";
const char *conv3_pmVar = "binData/conv3_pVar.bin";

const char *conv4_depth = "binData/conv4_d.bin";
const char *conv4_point = "binData/conv4_p.bin";
const char *conv4_dBeta = "binData/conv4_dbeta.bin";
const char *conv4_pBeta = "binData/conv4_pbeta.bin";
const char *conv4_dGamma = "binData/conv4_dgamma.bin";
const char *conv4_pGamma = "binData/conv4_pgamma.bin";
const char *conv4_dmMean = "binData/conv4_dMean.bin";
const char *conv4_dmVar = "binData/conv4_dVar.bin";
const char *conv4_pmMean = "binData/conv4_pMean.bin";
const char *conv4_pmVar = "binData/conv4_pVar.bin";

const char *conv5_depth = "binData/conv5_d.bin";
const char *conv5_point = "binData/conv5_p.bin";
const char *conv5_dBeta = "binData/conv5_dbeta.bin";
const char *conv5_pBeta = "binData/conv5_pbeta.bin";
const char *conv5_dGamma = "binData/conv5_dgamma.bin";
const char *conv5_pGamma = "binData/conv5_pgamma.bin";
const char *conv5_dmMean = "binData/conv5_dMean.bin";
const char *conv5_dmVar = "binData/conv5_dVar.bin";
const char *conv5_pmMean = "binData/conv5_pMean.bin";
const char *conv5_pmVar = "binData/conv5_pVar.bin";

const char *conv6_depth = "binData/conv6_d.bin";
const char *conv6_point = "binData/conv6_p.bin";
const char *conv6_dBeta = "binData/conv6_dbeta.bin";
const char *conv6_pBeta = "binData/conv6_pbeta.bin";
const char *conv6_dGamma = "binData/conv6_dgamma.bin";
const char *conv6_pGamma = "binData/conv6_pgamma.bin";
const char *conv6_dmMean = "binData/conv6_dMean.bin";
const char *conv6_dmVar = "binData/conv6_dVar.bin";
const char *conv6_pmMean = "binData/conv6_pMean.bin";
const char *conv6_pmVar = "binData/conv6_pVar.bin";

const char *conv7_depth = "binData/conv7_d.bin";
const char *conv7_point = "binData/conv7_p.bin";
const char *conv7_dBeta = "binData/conv7_dbeta.bin";
const char *conv7_pBeta = "binData/conv7_pbeta.bin";
const char *conv7_dGamma = "binData/conv7_dgamma.bin";
const char *conv7_pGamma = "binData/conv7_pgamma.bin";
const char *conv7_dmMean = "binData/conv7_dMean.bin";
const char *conv7_dmVar = "binData/conv7_dVar.bin";
const char *conv7_pmMean = "binData/conv7_pMean.bin";
const char *conv7_pmVar = "binData/conv7_pVar.bin";

const char *conv8_depth = "binData/conv8_d.bin";
const char *conv8_point = "binData/conv8_p.bin";
const char *conv8_dBeta = "binData/conv8_dbeta.bin";
const char *conv8_pBeta = "binData/conv8_pbeta.bin";
const char *conv8_dGamma = "binData/conv8_dgamma.bin";
const char *conv8_pGamma = "binData/conv8_pgamma.bin";
const char *conv8_dmMean = "binData/conv8_dMean.bin";
const char *conv8_dmVar = "binData/conv8_dVar.bin";
const char *conv8_pmMean = "binData/conv8_pMean.bin";
const char *conv8_pmVar = "binData/conv8_pVar.bin";

const char *conv9_depth = "binData/conv9_d.bin";
const char *conv9_point = "binData/conv9_p.bin";
const char *conv9_dBeta = "binData/conv9_dbeta.bin";
const char *conv9_pBeta = "binData/conv9_pbeta.bin";
const char *conv9_dGamma = "binData/conv9_dgamma.bin";
const char *conv9_pGamma = "binData/conv9_pgamma.bin";
const char *conv9_dmMean = "binData/conv9_dMean.bin";
const char *conv9_dmVar = "binData/conv9_dVar.bin";
const char *conv9_pmMean = "binData/conv9_pMean.bin";
const char *conv9_pmVar = "binData/conv9_pVar.bin";

const char *conv10_depth = "binData/conv10_d.bin";
const char *conv10_point = "binData/conv10_p.bin";
const char *conv10_dBeta = "binData/conv10_dbeta.bin";
const char *conv10_pBeta = "binData/conv10_pbeta.bin";
const char *conv10_dGamma = "binData/conv10_dgamma.bin";
const char *conv10_pGamma = "binData/conv10_pgamma.bin";
const char *conv10_dmMean = "binData/conv10_dMean.bin";
const char *conv10_dmVar = "binData/conv10_dVar.bin";
const char *conv10_pmMean = "binData/conv10_pMean.bin";
const char *conv10_pmVar = "binData/conv10_pVar.bin";

const char *conv11_depth = "binData/conv11_d.bin";
const char *conv11_point = "binData/conv11_p.bin";
const char *conv11_dBeta = "binData/conv11_dbeta.bin";
const char *conv11_pBeta = "binData/conv11_pbeta.bin";
const char *conv11_dGamma = "binData/conv11_dgamma.bin";
const char *conv11_pGamma = "binData/conv11_pgamma.bin";
const char *conv11_dmMean = "binData/conv11_dMean.bin";
const char *conv11_dmVar = "binData/conv11_dVar.bin";
const char *conv11_pmMean = "binData/conv11_pMean.bin";
const char *conv11_pmVar = "binData/conv11_pVar.bin";

const char *conv12_depth = "binData/conv12_d.bin";
const char *conv12_point = "binData/conv12_p.bin";
const char *conv12_dBeta = "binData/conv12_dbeta.bin";
const char *conv12_pBeta = "binData/conv12_pbeta.bin";
const char *conv12_dGamma = "binData/conv12_dgamma.bin";
const char *conv12_pGamma = "binData/conv12_pgamma.bin";
const char *conv12_dmMean = "binData/conv12_dMean.bin";
const char *conv12_dmVar = "binData/conv12_dVar.bin";
const char *conv12_pmMean = "binData/conv12_pMean.bin";
const char *conv12_pmVar = "binData/conv12_pVar.bin";

const char *conv13_depth = "binData/conv13_d.bin";
const char *conv13_point = "binData/conv13_p.bin";
const char *conv13_dBeta = "binData/conv13_dbeta.bin";
const char *conv13_pBeta = "binData/conv13_pbeta.bin";
const char *conv13_dGamma = "binData/conv13_dgamma.bin";
const char *conv13_pGamma = "binData/conv13_pgamma.bin";
const char *conv13_dmMean = "binData/conv13_dMean.bin";
const char *conv13_dmVar = "binData/conv13_dVar.bin";
const char *conv13_pmMean = "binData/conv13_pMean.bin";
const char *conv13_pmVar = "binData/conv13_pVar.bin";

const char *conv14_point = "binData/conv14_p.bin";
const char *conv14_pBeta = "binData/conv14_pbeta.bin";
const char *conv14_pGamma = "binData/conv14_pgamma.bin";
const char *conv14_pmMean = "binData/conv14_pMean.bin";
const char *conv14_pmVar = "binData/conv14_pVar.bin";
const char *conv14_w = "binData/conv14_2_d.bin";
const char *conv14_wBeta = "binData/conv14_2_dbeta.bin";
const char *conv14_wGamma = "binData/conv14_2_dgamma.bin";
const char *conv14_wmMean = "binData/conv14_2_dMean.bin";
const char *conv14_wmVar = "binData/conv14_2_dVar.bin";

const char *conv15_point = "binData/conv15_p.bin";
const char *conv15_pBeta = "binData/conv15_pbeta.bin";
const char *conv15_pGamma = "binData/conv15_pgamma.bin";
const char *conv15_pmMean = "binData/conv15_pMean.bin";
const char *conv15_pmVar = "binData/conv15_pVar.bin";
const char *conv15_w = "binData/conv15_2_d.bin";
const char *conv15_wBeta = "binData/conv15_2_dbeta.bin";
const char *conv15_wGamma = "binData/conv15_2_dgamma.bin";
const char *conv15_wmMean = "binData/conv15_2_dMean.bin";
const char *conv15_wmVar = "binData/conv15_2_dVar.bin";

const char *conv16_point = "binData/conv16_p.bin";
const char *conv16_pBeta = "binData/conv16_pbeta.bin";
const char *conv16_pGamma = "binData/conv16_pgamma.bin";
const char *conv16_pmMean = "binData/conv16_pMean.bin";
const char *conv16_pmVar = "binData/conv16_pVar.bin";
const char *conv16_w = "binData/conv16_2_d.bin";
const char *conv16_wBeta = "binData/conv16_2_dbeta.bin";
const char *conv16_wGamma = "binData/conv16_2_dgamma.bin";
const char *conv16_wmMean = "binData/conv16_2_dMean.bin";
const char *conv16_wmVar = "binData/conv16_2_dVar.bin";

const char *conv17_point = "binData/conv17_p.bin";
const char *conv17_pBeta = "binData/conv17_pbeta.bin";
const char *conv17_pGamma = "binData/conv17_pgamma.bin";
const char *conv17_pmMean = "binData/conv17_pMean.bin";
const char *conv17_pmVar = "binData/conv17_pVar.bin";
const char *conv17_w = "binData/conv17_2_d.bin";
const char *conv17_wBeta = "binData/conv17_2_dbeta.bin";
const char *conv17_wGamma = "binData/conv17_2_dgamma.bin";
const char *conv17_wmMean = "binData/conv17_2_dMean.bin";
const char *conv17_wmVar = "binData/conv17_2_dVar.bin";

const char *box0_loc_b = "binData/box0_boxp_b.bin";
const char *box0_loc_w = "binData/box0_boxp_w.bin";
const char *box0_conf_b = "binData/box0_classp_b.bin";
const char *box0_conf_w = "binData/box0_classp_w.bin";

const char *box1_loc_b = "binData/box1_boxp_b.bin";
const char *box1_loc_w = "binData/box1_boxp_w.bin";
const char *box1_conf_b = "binData/box1_classp_b.bin";
const char *box1_conf_w = "binData/box1_classp_w.bin";

const char *box2_loc_b = "binData/box2_boxp_b.bin";
const char *box2_loc_w = "binData/box2_boxp_w.bin";
const char *box2_conf_b = "binData/box2_classp_b.bin";
const char *box2_conf_w = "binData/box2_classp_w.bin";

const char *box3_loc_b = "binData/box3_boxp_b.bin";
const char *box3_loc_w = "binData/box3_boxp_w.bin";
const char *box3_conf_b = "binData/box3_classp_b.bin";
const char *box3_conf_w = "binData/box3_classp_w.bin";

const char *box4_loc_b = "binData/box4_boxp_b.bin";
const char *box4_loc_w = "binData/box4_boxp_w.bin";
const char *box4_conf_b = "binData/box4_classp_b.bin";
const char *box4_conf_w = "binData/box4_classp_w.bin";

const char *box5_loc_b = "binData/box5_boxp_b.bin";
const char *box5_loc_w = "binData/box5_boxp_w.bin";
const char *box5_conf_b = "binData/box5_classp_b.bin";
const char *box5_conf_w = "binData/box5_classp_w.bin";



OCR::OCR(){}

OCR::~OCR(){}

void OCR::init() {
	conv0.init_layer_standard(3, 32, 3, conv0_bin, bnScale0, bnBias0, conv0_mMean, conv0_mVar, NULL);
	conv1.init_layer_dsc(32, 64, 3, 1, conv1_depth, conv1_point, conv1_dGamma, conv1_dBeta, conv1_pGamma, conv1_pBeta, conv1_dmMean, conv1_dmVar, conv1_pmMean, conv1_pmVar, NULL);
	conv2.init_layer_dsc(64, 128, 3, 1, conv2_depth, conv2_point, conv2_dGamma, conv2_dBeta, conv2_pGamma, conv2_pBeta, conv2_dmMean, conv2_dmVar, conv2_pmMean, conv2_pmVar, NULL);
	conv3.init_layer_dsc(128, 128, 3, 1, conv3_depth, conv3_point, conv3_dGamma, conv3_dBeta, conv3_pGamma, conv3_pBeta, conv3_dmMean, conv3_dmVar, conv3_pmMean, conv3_pmVar, NULL);
	conv4.init_layer_dsc(128, 256, 3, 1, conv4_depth, conv4_point, conv4_dGamma, conv4_dBeta, conv4_pGamma, conv4_pBeta, conv4_dmMean, conv4_dmVar, conv4_pmMean, conv4_pmVar, NULL);
	conv5.init_layer_dsc(256, 256, 3, 1, conv5_depth, conv5_point, conv5_dGamma, conv5_dBeta, conv5_pGamma, conv5_pBeta, conv5_dmMean, conv5_dmVar, conv5_pmMean, conv5_pmVar, NULL);
	conv6.init_layer_dsc(256, 512, 3, 1, conv6_depth, conv6_point, conv6_dGamma, conv6_dBeta, conv6_pGamma, conv6_pBeta, conv6_dmMean, conv6_dmVar, conv6_pmMean, conv6_pmVar, NULL);
	conv7.init_layer_dsc(512, 512, 3, 1, conv7_depth, conv7_point, conv7_dGamma, conv7_dBeta, conv7_pGamma, conv7_pBeta, conv7_dmMean, conv7_dmVar, conv7_pmMean, conv7_pmVar, NULL);
	conv8.init_layer_dsc(512, 512, 3, 1, conv8_depth, conv8_point, conv8_dGamma, conv8_dBeta, conv8_pGamma, conv8_pBeta, conv8_dmMean, conv8_dmVar, conv8_pmMean, conv8_pmVar, NULL);
	conv9.init_layer_dsc(512, 512, 3, 1, conv9_depth, conv9_point, conv9_dGamma, conv9_dBeta, conv9_pGamma, conv9_pBeta, conv9_dmMean, conv9_dmVar, conv9_pmMean, conv9_pmVar, NULL);
	conv10.init_layer_dsc(512, 512, 3, 1, conv10_depth, conv10_point, conv10_dGamma, conv10_dBeta, conv10_pGamma, conv10_pBeta, conv10_dmMean, conv10_dmVar, conv10_pmMean, conv10_pmVar, NULL);
	conv11.init_layer_dsc(512, 512, 3, 1, conv11_depth, conv11_point, conv11_dGamma, conv11_dBeta, conv11_pGamma, conv11_pBeta, conv11_dmMean, conv11_dmVar, conv11_pmMean, conv11_pmVar, NULL);
	conv12.init_layer_dsc(512, 1024, 3, 1, conv12_depth, conv12_point, conv12_dGamma, conv12_dBeta, conv12_pGamma, conv12_pBeta, conv12_dmMean, conv12_dmVar, conv12_pmMean, conv12_pmVar, NULL);
	conv13.init_layer_dsc(1024, 1024, 3, 1, conv13_depth, conv13_point, conv13_dGamma, conv13_dBeta, conv13_pGamma, conv13_pBeta, conv13_dmMean, conv13_dmVar, conv13_pmMean, conv13_pmVar, NULL);
	conv14.init_layer_extra(1024, 256, 512, 3, 1, conv14_w, conv14_point, conv14_wGamma, conv14_wBeta, conv14_pGamma, conv14_pBeta, conv14_wmMean, conv14_wmVar, conv14_pmMean, conv14_pmVar, NULL);
	conv15.init_layer_extra(512, 128, 256, 3, 1, conv15_w, conv15_point, conv15_wGamma, conv15_wBeta, conv15_pGamma, conv15_pBeta, conv15_wmMean, conv15_wmVar, conv15_pmMean, conv15_pmVar, NULL);
	conv16.init_layer_extra(256, 128, 256, 3, 1, conv16_w, conv16_point, conv16_wGamma, conv16_wBeta, conv16_pGamma, conv16_pBeta, conv16_wmMean, conv16_wmVar, conv16_pmMean, conv16_pmVar, NULL);
	conv17.init_layer_extra(256, 64, 128, 3, 1, conv17_w, conv17_point, conv17_wGamma, conv17_wBeta, conv17_pGamma, conv17_pBeta, conv17_wmMean, conv17_wmVar, conv17_pmMean, conv17_pmVar, NULL);
	box0_loc.init_layer_box(512, 12, 1, box0_loc_w, box0_loc_b, NULL);
	box0_conf.init_layer_box(512, (num_classes + 1) * 3, 1, box0_conf_w, box0_conf_b, NULL);
	box1_loc.init_layer_box(1024, 24, 1, box1_loc_w, box1_loc_b, NULL);
	box1_conf.init_layer_box(1024, (num_classes + 1) * 6, 1, box1_conf_w, box1_conf_b, NULL);
	box2_loc.init_layer_box(512, 24, 1, box2_loc_w, box2_loc_b, NULL);
	box2_conf.init_layer_box(512, (num_classes + 1) * 6, 1, box2_conf_w, box2_conf_b, NULL);
	box3_loc.init_layer_box(256, 24, 1, box3_loc_w, box3_loc_b, NULL);
	box3_conf.init_layer_box(256, (num_classes + 1) * 6, 1, box3_conf_w, box3_conf_b, NULL);
	box4_loc.init_layer_box(256, 24, 1, box4_loc_w, box4_loc_b, NULL);
	box4_conf.init_layer_box(256, (num_classes + 1) * 6, 1, box4_conf_w, box4_conf_b, NULL);
	box5_loc.init_layer_box(128, 24, 1, box5_loc_w, box5_loc_b, NULL);
	box5_conf.init_layer_box(128, (num_classes + 1) * 6, 1, box5_conf_w, box5_conf_b, NULL);
	return;
}

void OCR::inference(Mat* input_image, Mat* output_image, string* result_word) {

	ssd_mobilenet.classify_example(input_image, output_image, result_word, conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
			conv13, conv14, conv15, conv16, conv17, box0_loc, box0_conf, box1_loc, box1_conf, box2_loc, box2_conf, box3_loc, box3_conf, box4_loc, box4_conf, box5_loc, box5_conf);
	return;
	//system("pause");
}