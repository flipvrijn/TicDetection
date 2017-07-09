#include <string>
#include "Detection.h"


Detection::Detection()
{
}


Detection::~Detection()
{
}

void Detection::enableFeedback()
{
	this->feedback = true;
}

bool Detection::feedbackEnabled()
{
	return this->feedback;
}