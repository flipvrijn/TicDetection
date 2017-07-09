#pragma once
#include "Tracker.h"

class FaceTracker : Tracker
{
private:
public:
	FaceTracker();
	FaceTracker(cv::Size size);
	~FaceTracker();

	cv::Rect getFaceRectangle();
	cv::Mat getFaceROI();
};