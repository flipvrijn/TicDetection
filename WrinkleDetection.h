#pragma once
#include "Detection.h"
class WrinkleDetection : public Detection
{
private:
	cv::String edgesWindowName;
	cv::Mat ROI;
	cv::Rect faceRectangle;
	cv::Mat detectedEdges;
	cv::Rect detectionRegion;
	int lowThreshold = 40;
	int maxThreshold = 100;
	int numNonZero;
	static void callback(int, void*);
public:
	WrinkleDetection();
	WrinkleDetection(cv::String edgesWindowName);
	~WrinkleDetection(); 

	void setWindowName(cv::String edgesWindowName);
	void setDetectionRegion(cv::Mat face, cv::Rect region);
	void setThreshold(int threshold);
	int getNumNonZero();
	cv::Mat getDetectedEdges();
	bool detect();
	void draw();
};

