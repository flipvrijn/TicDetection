#pragma once
#include "flandmark_detector.h"
#include "CascadeDetection.h"
class FaceDetection : public CascadeDetection
{
private:
	FLANDMARK_Model *flandmarkModel = flandmark_init("flandmark_model.dat");
	double *facialLandmarks = (double*)malloc(2 * flandmarkModel->data.options.M*sizeof(double));
	void deskew(cv::Mat);
public:
	FaceDetection();
	FaceDetection(cv::String, cv::Size);
	~FaceDetection();
	void preprocess();
	std::vector<cv::Point> getLandmarks();
};

