#pragma once
#include "Feature.h"
class MotionFeature : public Feature
{
public:
	MotionFeature(int);
	~MotionFeature();
	void calculateValue(std::vector<float>);
};

