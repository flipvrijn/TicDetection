#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CascadeDetection.h"
#include "FaceDetection.h"
#include "flandmark_detector.h"

using namespace cv;

FaceDetection::FaceDetection()
{
}

FaceDetection::FaceDetection(String cascadeFile, Size size)
{
	assert(this->cascadeClassifier.load(cascadeFile));
	this->size = size;
}

FaceDetection::~FaceDetection()
{
}

void FaceDetection::preprocess()
{
	Mat image = this->getImage().clone();

	//this->deskew(image);

	vector<Mat> rgbChannels(3);
	split(image, rgbChannels);
	IplImage imgGray = rgbChannels[2];

	Rect faceRect = this->getRect();

	int bbox[] = { faceRect.x, faceRect.y, faceRect.x + faceRect.width, faceRect.y + faceRect.height };

	flandmark_detect(&imgGray, bbox, flandmarkModel, facialLandmarks);

	// Update ROI
	Mat tmp;
	image.copyTo(tmp);
	this->ROI = tmp(faceRect);
}

void FaceDetection::deskew(Mat image)
{
	vector<Mat> rgbChannels(3);
	split(image, rgbChannels);
	IplImage imgGray = rgbChannels[2];

	Rect faceRect = this->getRect();

	int bbox[] = { faceRect.x, faceRect.y, faceRect.x + faceRect.width, faceRect.y + faceRect.height };

	flandmark_detect(&imgGray, bbox, flandmarkModel, facialLandmarks);

	Point leftEye(facialLandmarks[10], facialLandmarks[11]);
	Point rightEye(facialLandmarks[12], facialLandmarks[13]);

	float angle = atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180.0 / 3.1415926;

	Point center(image.cols / 2, image.rows / 2);

	Mat rotMat = getRotationMatrix2D(center, angle, 1);
	warpAffine(image, image, rotMat, image.size());
}

vector<Point> FaceDetection::getLandmarks()
{
	vector<Point> landmarks;

	for (int i = 2; i < 2 * flandmarkModel->data.options.M; i += 2)
	{
		landmarks.push_back(Point(facialLandmarks[i], facialLandmarks[i + 1]));
	}
	return landmarks;
}