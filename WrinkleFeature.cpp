#include "WrinkleFeature.h"

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

WrinkleFeature::WrinkleFeature(int type)
{
	this->setType(type);
}


WrinkleFeature::~WrinkleFeature()
{
}

void WrinkleFeature::calculateValue(std::vector<WrinkleData> slidingWindowData)
{
	int sum = 0;
	for (size_t i = 0; i < slidingWindowData.size(); i++)
	{
		WrinkleData wrinkleData = slidingWindowData[i];
		sum += countNonZero(wrinkleData.frame);
	}
	this->setValue(sum / slidingWindowData.size());
}