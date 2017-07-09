#include <vector>
#include <iostream>
#include "DistanceFeature.h"

using namespace std;


DistanceFeature::DistanceFeature(DistMeasurements distMeasurement, int featureType)
{
	this->distMeasurement = distMeasurement;
	this->setType(featureType);
}


DistanceFeature::~DistanceFeature()
{
}

void DistanceFeature::calculateValue(DistanceData distanceData)
{
	float distance;
	switch (this->distMeasurement)
	{
	case EUCLEDIAN:
		distance = static_cast<float>(sqrt(pow((distanceData.pointOne.x - distanceData.pointTwo.x), 2) + pow((distanceData.pointOne.y - distanceData.pointTwo.y), 2)));
		break;
	case MANHATTAN:
		distance = static_cast<float>((distanceData.pointOne.x - distanceData.pointTwo.x) + (distanceData.pointOne.y - distanceData.pointTwo.y));
		break;
	default:
		throw runtime_error("Error: Unknown distance measurement!");
	}

	this->setValue(distance / distanceData.relativeTo);
}