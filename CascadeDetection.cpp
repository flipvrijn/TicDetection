#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CascadeDetection.h"

using namespace cv;


CascadeDetection::CascadeDetection()
{
}

CascadeDetection::CascadeDetection(String cascadeFile)
{
	this->setFile(cascadeFile);
}

CascadeDetection::CascadeDetection(String cascadeFile, Size size)
{
	this->setFile(cascadeFile);
	this->setSize(size);
}


CascadeDetection::~CascadeDetection()
{
}


bool CascadeDetection::detect(Mat frame)
{
	this->image = frame;

	vector<Rect> objects;

	vector<Mat> rgbChannels(3);
	split(frame, rgbChannels);
	Mat frame_gray = rgbChannels[2];
	Mat tmp;

	this->cascadeClassifier.detectMultiScale(frame_gray, objects, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, this->size);

	if (objects.size() > 0)
	{
		this->rect = objects[0];
		frame.copyTo(tmp);
		this->ROI = tmp(rect);
		return true;
	}

	return false;
}

void CascadeDetection::setSize(Size size)
{
	this->size = size;
}

void CascadeDetection::setFile(String file)
{
	assert(this->cascadeClassifier.load(file));
}

Mat CascadeDetection::getImage()
{
	return this->image.clone();
}

Rect CascadeDetection::getRect()
{
	return this->rect;
}

Mat CascadeDetection::getROI()
{
	return this->ROI;
}
