#include <ctime>

#include "Feature.h"

Feature::Feature()
{
}

Feature::~Feature()
{
}

void Feature::setType(int type)
{
	this->empty = false;
	this->type = type;
}

void Feature::setValue(float value)
{
	this->empty = false;
	this->value = value;
}

int Feature::getType()
{
	return this->type;
}

float Feature::getValue()
{
	return this->value;
}

bool Feature::isEmpty()
{
	return this->empty;
}