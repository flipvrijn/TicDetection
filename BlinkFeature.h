#pragma once
#include "BlinkData.h"
#include "Feature.h"
class BlinkFeature :
	public Feature
{
public:
	BlinkFeature(int);
	~BlinkFeature();
	void calculateValue(std::vector<BlinkData> slidingWindowData);
};

