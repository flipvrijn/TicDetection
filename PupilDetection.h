#pragma once
#include "Eye.h"
#include "Detection.h"
class PupilDetection : public Detection
{
private:
	cv::Mat faceROI;
	cv::Rect faceRectangle;
	cv::String pupilWindowName;
	std::vector<Eye> eyes;
	std::vector<Eye> findEyes();
	cv::Rect leftRightCornerRegion;
	cv::Rect leftLeftCornerRegion;
	cv::Rect rightRightCornerRegion;
	cv::Rect rightLeftCornerRegion;
public:
	PupilDetection();
	PupilDetection(cv::String pupilWindowName);
	~PupilDetection();
	void setWindowName(cv::String pupilWindowName);
	void setFaceROI(cv::Mat faceROI);
	void setFaceRectangle(cv::Rect face);
	bool detect();
	std::vector<cv::Rect> getEyeRegions();
	std::vector<Eye> getPupils();
	void draw();
};

