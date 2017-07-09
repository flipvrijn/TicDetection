#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "Eye.h"
#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include "PupilDetection.h"
#include "Detection.h"

using namespace std;
using namespace cv;

PupilDetection::PupilDetection()
{
}

PupilDetection::PupilDetection(String pupilWindowName)
{
	this->setWindowName(pupilWindowName);
}

PupilDetection::~PupilDetection()
{
}

void PupilDetection::setFaceROI(Mat faceROI)
{
	Mat tmp;
	faceROI.copyTo(tmp);
	this->faceROI = tmp;
}

void PupilDetection::setFaceRectangle(Rect face)
{
	this->faceRectangle = face;
}

void PupilDetection::setWindowName(String pupilWindowName)
{
	if (this->feedbackEnabled())
	{
		this->pupilWindowName = pupilWindowName;

		namedWindow(pupilWindowName, CV_WINDOW_NORMAL);
		resizeWindow(pupilWindowName, 250, 250);
		moveWindow(pupilWindowName, 0, 0);
	}
}

vector<Rect> PupilDetection::getEyeRegions()
{
	int eye_region_width = (int)(this->faceRectangle.width * (kEyePercentWidth / 100.0));
	int eye_region_height = (int)(this->faceRectangle.width * (kEyePercentHeight / 100.0));
	int eye_region_top = (int)(this->faceRectangle.height * (kEyePercentTop / 100.0));
	Rect leftEyeRegion((int)(this->faceRectangle.width*(kEyePercentSide / 100.0)), eye_region_top, eye_region_width, eye_region_height);
	Rect rightEyeRegion((int)(this->faceRectangle.width - eye_region_width - this->faceRectangle.width*(kEyePercentSide / 100.0)), eye_region_top, eye_region_width, eye_region_height);

	vector<Rect> regions;
	regions.push_back(leftEyeRegion);
	regions.push_back(rightEyeRegion);

	return regions;
}

vector<Eye> PupilDetection::findEyes()
{
	vector<Mat> rgbChannels(3);
	split(this->faceROI, rgbChannels);
	Mat frameGray = rgbChannels[2];

	if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * this->faceRectangle.width;
		GaussianBlur(frameGray, frameGray, Size(0, 0), sigma);
	}

	//-- Find eye regions and draw them
	vector<Rect> regions = this->getEyeRegions();
	Rect leftEyeRegion = regions[0];
	Rect rightEyeRegion = regions[1];

	//-- Find Eye Centers
	Point leftPupil = findEyeCenter(frameGray, leftEyeRegion, "Left Eye");
	Point rightPupil = findEyeCenter(frameGray, rightEyeRegion, "Right Eye");
	// get corner regions
	this->leftRightCornerRegion.x = leftEyeRegion.x;
	this->leftRightCornerRegion.y = leftEyeRegion.y;
	this->leftRightCornerRegion.width = leftEyeRegion.width;
	this->leftRightCornerRegion.width -= leftPupil.x;
	this->leftRightCornerRegion.x += leftPupil.x;
	this->leftRightCornerRegion.height = leftEyeRegion.height;
	this->leftRightCornerRegion.height /= 2;
	this->leftRightCornerRegion.y += this->leftRightCornerRegion.height / 2;
	this->leftLeftCornerRegion.x = leftEyeRegion.x;
	this->leftLeftCornerRegion.y = leftEyeRegion.y;
	this->leftLeftCornerRegion.width = leftPupil.x;
	this->leftLeftCornerRegion.height = leftEyeRegion.height;
	this->leftLeftCornerRegion.height /= 2;
	this->leftLeftCornerRegion.y += this->leftLeftCornerRegion.height / 2;
	this->rightLeftCornerRegion.x = rightEyeRegion.x;
	this->rightLeftCornerRegion.y = rightEyeRegion.y;
	this->rightLeftCornerRegion.width = rightPupil.x;
	this->rightLeftCornerRegion.height = rightEyeRegion.height;
	this->rightLeftCornerRegion.height /= 2;
	this->rightLeftCornerRegion.y += this->rightLeftCornerRegion.height / 2;
	this->rightRightCornerRegion.x = rightEyeRegion.x;
	this->rightRightCornerRegion.y = rightEyeRegion.y;
	this->rightRightCornerRegion.width = rightEyeRegion.width;
	this->rightRightCornerRegion.width -= rightPupil.x;
	this->rightRightCornerRegion.x += rightPupil.x;
	this->rightRightCornerRegion.height = rightEyeRegion.height;
	this->rightRightCornerRegion.height /= 2;
	this->rightRightCornerRegion.y += this->rightRightCornerRegion.height / 2;
	
	// Change eye centers to face coordinates
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;

	//-- Find Eye Corners
	if (kEnableEyeCorner) {
		Point2f leftRightCorner = findEyeCorner(frameGray(this->leftRightCornerRegion), true, false);
		leftRightCorner.x += this->leftRightCornerRegion.x;
		leftRightCorner.y += this->leftRightCornerRegion.y;
		Point2f leftLeftCorner = findEyeCorner(frameGray(this->leftRightCornerRegion), true, true);
		leftLeftCorner.x += this->leftRightCornerRegion.x;
		leftLeftCorner.y += this->leftRightCornerRegion.y;
		Point2f rightLeftCorner = findEyeCorner(frameGray(this->leftRightCornerRegion), false, true);
		rightLeftCorner.x += this->leftRightCornerRegion.x;
		rightLeftCorner.y += this->leftRightCornerRegion.y;
		Point2f rightRightCorner = findEyeCorner(frameGray(this->leftRightCornerRegion), false, false);
		rightRightCorner.x += this->leftRightCornerRegion.x;
		rightRightCorner.y += this->leftRightCornerRegion.y;
		circle(frameGray, leftRightCorner, 3, 200);
		circle(frameGray, leftLeftCorner, 3, 200);
		circle(frameGray, rightLeftCorner, 3, 200);
		circle(frameGray, rightRightCorner, 3, 200);
	}

	vector<Eye> eyes;
	Eye leftEye, rightEye;

	// See if the pupils are visible (blinking / keeping eyes shut)
	if (leftPupil.y < this->leftRightCornerRegion.y || leftPupil.y > this->leftRightCornerRegion.y + this->leftRightCornerRegion.height)
	{
		leftEye.pupilVisible = false;
	}

	if (rightPupil.y < this->leftRightCornerRegion.y || rightPupil.y > this->leftRightCornerRegion.y + this->leftRightCornerRegion.height)
	{
		rightEye.pupilVisible = false;
	}

	// Storing the information
	leftEye.pupil = leftPupil;
	rightEye.pupil = rightPupil;

	eyes.push_back(leftEye);
	eyes.push_back(rightEye);

	return eyes;
}

bool PupilDetection::detect()
{
	createCornerKernels();

	//-- Show what you got
	if (faceRectangle.width > 0 && faceRectangle.height > 0 && !faceROI.empty()) {
		this->eyes = this->findEyes();		

		if (this->feedbackEnabled())
			this->draw();

		return true;
	}

	return false;
}

void PupilDetection::draw()
{
	// Display eye regions
	rectangle(this->faceROI, this->leftLeftCornerRegion, Scalar(255, 255, 255));
	rectangle(this->faceROI, this->leftRightCornerRegion, Scalar(255, 255, 255));
	rectangle(this->faceROI, this->rightLeftCornerRegion, Scalar(255, 255, 255));
	rectangle(this->faceROI, this->rightRightCornerRegion, Scalar(255, 255, 255));

	// Display eyes
	for (size_t i = 0; i < this->eyes.size(); i++)
	{
		Eye eye = this->eyes[i];
		Scalar pupilColor = Scalar(255, 255, 255);
		if (!eye.pupilVisible)
			pupilColor = Scalar(0, 0, 255);
		circle(this->faceROI, Point(eye.pupil.x, eye.pupil.y), 3, pupilColor);
	}

	imshow(pupilWindowName, this->faceROI);
}

vector<Eye> PupilDetection::getPupils()
{
	return this->eyes;
}