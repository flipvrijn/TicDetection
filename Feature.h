#pragma once
#include <ctime>
class Feature
{
private:
	float value = 0.0;
	bool empty = true;
	int type;
public:
	Feature();
	~Feature();
	void setType(int);
	int getType();
	void setValue(float);
	float getValue();
	int getThreshold();
	bool isEmpty();
};

