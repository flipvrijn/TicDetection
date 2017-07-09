#include <iostream>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "WrinkleDetection.h"

using namespace cv;

WrinkleDetection::WrinkleDetection()
{
}

WrinkleDetection::WrinkleDetection(String edgesWindowName)
{
	this->setWindowName(edgesWindowName);
}


WrinkleDetection::~WrinkleDetection()
{
}

void WrinkleDetection::setWindowName(String edgesWindowName)
{
	if (this->feedbackEnabled())
	{
		this->edgesWindowName = edgesWindowName;
		namedWindow(this->edgesWindowName, CV_WINDOW_NORMAL);
		resizeWindow(this->edgesWindowName, 250, 250);
		moveWindow(this->edgesWindowName, 0, 500);
	}
}

void WrinkleDetection::setDetectionRegion(Mat face, Rect region)
{
	Mat tmp;
	face.copyTo(tmp);
	this->ROI = tmp(region);
	this->detectionRegion = region;
}

void WrinkleDetection::setThreshold(int threshold)
{
	this->lowThreshold = threshold;
}

int WrinkleDetection::getNumNonZero()
{
	return this->numNonZero;
}

Mat WrinkleDetection::getDetectedEdges()
{
	return this->detectedEdges;
}

void WrinkleDetection::callback(int, void* object)
{
	WrinkleDetection* wd = (WrinkleDetection*)object;

	vector<Mat> rgbChannels(3);
	split(wd->ROI, rgbChannels);
	Mat frameGray = rgbChannels[2];

	// Reduce noise with a kernel 3x3
	blur(frameGray, wd->detectedEdges, Size(3, 3));
	Canny(wd->detectedEdges, wd->detectedEdges, wd->lowThreshold, wd->lowThreshold * 3, 3);

	wd->numNonZero = countNonZero(wd->detectedEdges);

	if (wd->feedbackEnabled())
		imshow(wd->edgesWindowName, wd->detectedEdges);
}

bool WrinkleDetection::detect()
{
	if (this->feedbackEnabled())
		createTrackbar("Threshold:", this->edgesWindowName, &this->lowThreshold, maxThreshold, &this->callback, this);

	// Do canny stuff
	if (!this->ROI.empty())
	{
		// Draw result
		this->callback(0, this);

		return true;
	}

	return false;
}