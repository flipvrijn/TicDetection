#include <vector>
#include "BlinkFeature.h"

using namespace std;


BlinkFeature::BlinkFeature(int type)
{
	this->setType(type);
}


BlinkFeature::~BlinkFeature()
{
}

void BlinkFeature::calculateValue(vector<BlinkData> slidingWindowData)
{
	float sumBlinks = 0.0;
	for (size_t i = 0; i < slidingWindowData.size(); i++)
	{
		BlinkData data = slidingWindowData[i];
		sumBlinks += data.blink;
	}

	this->setValue(sumBlinks / static_cast<float>(slidingWindowData.size()));
}