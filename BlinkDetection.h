#pragma once
#include "Detection.h"
class BlinkDetection : public Detection
{
private:
	cv::String windowName;
	cv::Rect faceRectangle;
	cv::Mat faceROI, result;
	std::vector<cv::Mat> eyeTemplates;
	double minVal, maxVal; cv::Point minLoc, maxLoc;
	float blinkThreshold;
	int currentEyeTemplate;
public:
	enum Eye{LEFT, RIGHT, NONE};
	BlinkDetection();
	BlinkDetection(cv::String windowName);
	~BlinkDetection();
	bool detect();
	void draw();
	void reset();
	void setWindowName(cv::String);
	void setFaceROI(cv::Mat);
	void setFaceRectangle(cv::Rect);
	void setEyeTemplate(cv::Mat);
	void setBlinkThreshold(float);
	float getMaxValue();
	std::pair<cv::Rect, cv::Rect> getEyeRegions();
};