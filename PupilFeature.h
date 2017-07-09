#pragma once
#include "Eye.h"
#include "Feature.h"
#include "PupilData.h"

class PupilFeature  :
	public Feature
{
public:
	PupilFeature(int);
	~PupilFeature();
	void calculateValue(std::vector<PupilData> slidingWindowData, int side);
	int getThreshold();
};

