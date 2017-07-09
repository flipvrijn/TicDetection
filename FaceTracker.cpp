#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv/cv.h"
#include <iostream>

#include "Tracker.h"
#include "FaceTracker.h"

using namespace cv;

FaceTracker::FaceTracker()
{
	this->hsize = 16;
	this->hranges[0] = 0;
	this->hranges[1] = 180;
	this->vmin = 10;
	this->vmax = 256;
	this->smin = 30;
	this->phranges = hranges;
}


FaceTracker::~FaceTracker()
{
}




Rect FaceTracker::getFaceRectangle()
{
	return this->prevRect;
}

Mat FaceTracker::getFaceROI()
{
	return this->image(this->prevRect);
}