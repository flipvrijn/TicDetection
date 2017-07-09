#pragma once
#include "Feature.h"
#include "WrinkleData.h"

class WrinkleFeature :
	public Feature
{
public:
	WrinkleFeature(int);
	~WrinkleFeature();
	void calculateValue(std::vector<WrinkleData> slidingWindowData);
	int getThreshold();
};


