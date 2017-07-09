#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Tracker.h"

using namespace cv;

Tracker::Tracker()
{
	this->hsize = 16;
	this->hranges[0] = 0;
	this->hranges[1] = 180;
	this->vmin = 10;
	this->vmax = 256;
	this->smin = 30;
	this->phranges = hranges;
}


Tracker::~Tracker()
{
}

void Tracker::setSize(Size size)
{
	this->HSVImg = Mat::zeros(size, CV_8UC3);
	this->hueImg = Mat::zeros(size, CV_8UC3);
	this->mask = Mat::zeros(size, CV_8UC3);
	this->probImg = Mat::zeros(size, CV_8UC1);
}

void Tracker::startTracking(Mat image, Rect rect)
{
	this->tracking = true;
	this->updateHueImage(image);

	Mat roi(this->hueImg, rect);
	Mat maskRoi(this->mask, rect);
	calcHist(&roi, 1, 0, maskRoi, this->hist, 1, &this->hsize, &this->phranges);
	normalize(this->hist, this->hist, 0, 255, CV_MINMAX);

	this->image = image;
	this->prevRect = rect;
}

void Tracker::track(Mat image)
{
	this->updateHueImage(image);

	calcBackProject(&this->hueImg, 1, 0, this->hist, this->probImg, &this->phranges);
	this->probImg &= mask;
	Size size = this->probImg.size();

	if (this->prevRect.x < 0) this->prevRect.x = 0;
	if (this->prevRect.x >= size.width) this->prevRect.x = size.width - 1;
	if (this->prevRect.y < 0) this->prevRect.y = 0;
	if (this->prevRect.y >= size.height) this->prevRect.y = size.height - 1;
	if (this->prevRect.x + this->prevRect.width > size.width) this->prevRect.width = size.width - this->prevRect.x;
	if (this->prevRect.y + this->prevRect.height > size.height) this->prevRect.height = size.height - this->prevRect.y;

	RotatedRect trackBox = CamShift(this->probImg, this->prevRect, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
	this->prevRect = trackBox.boundingRect();
	if (this->prevRect.area() <= 1)
	{
		int cols = this->probImg.cols, rows = this->probImg.rows, r = (MIN(cols, rows) + 5) / 6;
		this->prevRect = Rect(this->prevRect.x - r, this->prevRect.y - r,
			this->prevRect.x + r, this->prevRect.y + r) &
			Rect(0, 0, cols, rows);
	}

	ellipse(image, trackBox, Scalar(0, 0, 255), 3, CV_AA);
	Point2f vertices[4];
	trackBox.points(vertices);
	for (int i = 0; i < 4; i++)
		line(image, vertices[i], vertices[(i + 1) % 4], Scalar(255, 255, 255));
	//rectangle(image, this->prevRect, Scalar(255, 255, 255), 1, CV_AA);

	this->image = image;
}

void Tracker::updateHueImage(Mat image)
{
	cvtColor(image, this->HSVImg, CV_BGR2HSV);
	int _vmin = this->vmin, _vmax = this->vmax;
	inRange(this->HSVImg, Scalar(0, this->smin, MIN(_vmin, _vmax)),
		Scalar(180, 256, MAX(_vmin, _vmax)), this->mask);

	int ch[] = { 0, 0 };
	this->hueImg.create(this->HSVImg.size(), this->HSVImg.depth());
	mixChannels(&this->HSVImg, 1, &this->hueImg, 1, ch, 1);
}

bool Tracker::isTracking()
{
	return this->tracking;
}

Rect Tracker::getRect()
{
	return this->prevRect;
}