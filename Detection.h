#pragma once
class Detection
{
private:
	bool feedback = false;
public:
	Detection();
	~Detection();
	bool detect();
	void enableFeedback();
	bool feedbackEnabled();
};