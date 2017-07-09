#include <vector>
#include <iostream>

#include "SlidingWindow.h"
#include "Eye.h"
#include "PupilFeature.h"

using namespace std;

PupilFeature::PupilFeature(int type)
{
	this->setType(type);
}


PupilFeature::~PupilFeature()
{
}

void PupilFeature::calculateValue(vector<PupilData> slidingWindowData, int side)
{
	double numOfNoPupil = 0.0;
	for (size_t i = 0; i < slidingWindowData.size(); i++)
	{
		PupilData data = slidingWindowData[i];
		Eye eye;
		switch (side)
		{
		case 0:
			eye = data.leftPupil;
			break;
		case 1:
			eye = data.rightPupil;
			break;
		default:
			throw runtime_error("Error: Unknown pupil side. 0 for 'left', 1 for 'right'");
		}

		if (!eye.pupilVisible)
		{
			numOfNoPupil++;
		}
	}
	this->setValue(numOfNoPupil / slidingWindowData.size());
}

int PupilFeature::getThreshold()
{
	return 15;
}