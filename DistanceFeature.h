#pragma once
#include "DistanceData.h"
#include "Feature.h"
class DistanceFeature :
	public Feature
{
public:
	enum DistMeasurements {
		EUCLEDIAN,
		MANHATTAN
	};
	DistanceFeature(DistMeasurements, int);
	~DistanceFeature();
	void calculateValue(DistanceData);
private:
	DistMeasurements distMeasurement;
};

