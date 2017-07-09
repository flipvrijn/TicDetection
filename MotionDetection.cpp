#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "MotionDetection.h"

using namespace cv;


MotionDetection::MotionDetection()
{
}


MotionDetection::MotionDetection(String windowName)
{
	this->setWindowName(windowName);
}


MotionDetection::~MotionDetection()
{
}

void MotionDetection::setWindowName(String windowName)
{
	if (this->feedbackEnabled())
	{
		this->motionWindowName = windowName;

		namedWindow(motionWindowName, CV_WINDOW_NORMAL);
		resizeWindow(motionWindowName, 250, 250);
		moveWindow(motionWindowName, 0, 255);
	}
}

void MotionDetection::setRect(Rect faceRect)
{
	this->faceRect = faceRect;
}

void MotionDetection::setEyeRect(Rect eyeRect)
{
	this->eyeRect = eyeRect;
}

void MotionDetection::reset()
{
	this->frames.clear();
	this->framesIndex = 0;
}

void MotionDetection::feedFrame(Mat frame)
{
	if (this->frames.size() < 2)
	{
		this->frames.push_back(frame.clone());
	}
	else
	{
		this->frames[this->framesIndex] = frame.clone();
	}

	this->framesIndex++;
	this->framesIndex = this->framesIndex % 2;
}

bool MotionDetection::isFilled()
{
	return (this->frames.size() == 2);
}

Mat MotionDetection::getDiff(Rect roi)
{
	return this->diff(roi);
}

bool MotionDetection::detect()
{
	if (this->frames.size() == 2)
	{
		Mat frameOne = this->frames[0];
		Mat frameTwo = this->frames[1];

		if (frameOne.cols == frameTwo.cols && frameOne.rows == frameTwo.rows)
		{
			absdiff(this->frames[0], this->frames[1], this->diff);

			vector<Mat> channels;
			split(this->diff, channels);
			this->diff = channels[2].clone();

			filter2D(this->diff, this->diff, -1, getGaussianKernel(3, -1));
			threshold(this->diff, this->diff, 30, 255, THRESH_BINARY);

			if (this->feedbackEnabled())
			{
				Mat diffCopy = this->diff.clone();
				diffCopy = diffCopy(this->faceRect);
				rectangle(diffCopy, this->eyeRect, Scalar(255,255,255));
				imshow(this->motionWindowName, diffCopy);
			}

			return true;
		}

		return false;
	}

	return false;
}