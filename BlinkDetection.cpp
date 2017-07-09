#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "BlinkDetection.h"
#include <iostream>
#include <utility>

using namespace std;
using namespace cv;


BlinkDetection::BlinkDetection()
{
}


BlinkDetection::~BlinkDetection()
{
}

void BlinkDetection::setWindowName(String windowName)
{
	if (this->feedbackEnabled())
	{
		this->windowName = windowName;
		resizeWindow(windowName, 250, 250);
		moveWindow(windowName, 250, 0);

		namedWindow(windowName);
	}
}

void BlinkDetection::setFaceROI(Mat faceROI)
{
	this->faceROI = faceROI.clone();

	this->result.create(faceROI.rows, faceROI.cols, CV_32FC1);
}

void BlinkDetection::setFaceRectangle(Rect face)
{
	this->faceRectangle = face;
}

void BlinkDetection::setEyeTemplate(Mat temp)
{
	if (this->eyeTemplates.size() < 2)
		eyeTemplates.push_back(temp);
}

void BlinkDetection::setBlinkThreshold(float threshold)
{
	this->blinkThreshold = threshold;
}

pair<Rect, Rect> BlinkDetection::getEyeRegions()
{
	const int kEyePercentTop = 25;
	const int kEyePercentSide = 13;
	const int kEyePercentHeight = 30;
	const int kEyePercentWidth = 35;

	int eye_region_width = (int)(this->faceRectangle.width * (kEyePercentWidth / 100.0));
	int eye_region_height = (int)(this->faceRectangle.width * (kEyePercentHeight / 100.0));
	int eye_region_top = (int)(this->faceRectangle.height * (kEyePercentTop / 100.0));
	Rect leftEyeRegion((int)(this->faceRectangle.width*(kEyePercentSide / 100.0)), eye_region_top, eye_region_width, eye_region_height);
	Rect rightEyeRegion((int)(this->faceRectangle.width - eye_region_width - this->faceRectangle.width*(kEyePercentSide / 100.0)), eye_region_top, eye_region_width, eye_region_height);

	pair<Rect, Rect> regions(leftEyeRegion, rightEyeRegion);

	return regions;
}

bool BlinkDetection::detect()
{
	if (this->eyeTemplates[currentEyeTemplate].rows == 0 || this->eyeTemplates[currentEyeTemplate].cols == 0)
		return false;

	matchTemplate(this->faceROI, this->eyeTemplates[currentEyeTemplate], this->result, CV_TM_CCOEFF_NORMED);
	minMaxLoc(this->result, &this->minVal, &this->maxVal, &this->minLoc, &this->maxLoc, Mat());

	if (this->feedbackEnabled())
		this->draw();

	this->currentEyeTemplate = (this->currentEyeTemplate + 1) % 2;

	if (this->result.at<float>(this->maxLoc) < this->blinkThreshold)
		return true;

	return false;
}

float BlinkDetection::getMaxValue()
{
	if (this->result.rows == 0)
		return 0.0;
	return this->result.at<float>(this->maxLoc);
}

void BlinkDetection::draw()
{
	pair<Rect, Rect> eyeRegions = this->getEyeRegions();
	rectangle(this->faceROI, eyeRegions.first, Scalar(255, 255, 255));
	rectangle(this->faceROI, eyeRegions.second, Scalar(255, 255, 255));

	if (this->result.at<float>(maxLoc) < this->blinkThreshold)
	{
		circle(this->faceROI, this->maxLoc, 3, Scalar(0, 0, 255));
		circle(this->result, this->maxLoc, 3, Scalar::all(0));
	}
	imwrite("result.png", this->result);
	imshow("result", this->result);

	imshow("Eye template", this->eyeTemplates[currentEyeTemplate]);
	imshow(this->windowName, this->faceROI);
}

void BlinkDetection::reset()
{
	this->result.create(faceROI.rows, faceROI.cols, CV_32FC1);
	this->result = Mat();
	this->eyeTemplates.clear();
}