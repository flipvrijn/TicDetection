#pragma once
#include "Detection.h"
class MotionDetection : public Detection
{
private:
	cv::String motionWindowName;
	std::vector<cv::Mat> frames;
	cv::Mat diff;
	int framesIndex = 0;
	cv::Rect faceRect;
	cv::Rect eyeRect;
public:
	MotionDetection();
	MotionDetection(cv::String);
	~MotionDetection();
	void setWindowName(cv::String);
	void setRect(cv::Rect);
	void setEyeRect(cv::Rect);
	bool isFilled();
	bool detect();
	void feedFrame(cv::Mat);
	cv::Mat getDiff(cv::Rect);
	void reset();
};

