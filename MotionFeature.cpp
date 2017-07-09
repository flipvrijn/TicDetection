#include <vector>
#include <iostream>
#include "MotionFeature.h"

using namespace std;

MotionFeature::MotionFeature(int type)
{
	this->setType(type);
}


MotionFeature::~MotionFeature()
{
}

void MotionFeature::calculateValue(vector<float> v)
{
	float sum = 0.0;
	for (size_t i = 0; i < v.size(); i++)
	{
		sum += v[i];
	}
	float value = sum / static_cast<float>(v.size());

	this->setValue(value);
}