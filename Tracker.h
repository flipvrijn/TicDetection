#pragma once
class Tracker
{
protected:
	void updateHueImage(cv::Mat image);

	float hranges[2];  // Histogram Range
	const float* phranges;
	int vmin, vmax, smin;
	int hsize;

	cv::Mat image;
	cv::Mat HSVImg;     // Image converted to HSV color mode
	cv::Mat hueImg;     // Hue channel of the HSV image
	cv::Mat mask;       // Image for masking pixels
	cv::Mat probImg;    // Face Probability Estimates for each pixel
	cv::Mat hist;
	cv::Mat HistImage;

	cv::Rect prevRect;        // Current Face-Location Estimate
	bool tracking = false;
public:
	Tracker();
	~Tracker();
	void setSize(cv::Size size);
	void startTracking(cv::Mat image, cv::Rect cvRect);
	void track(cv::Mat image);
	cv::Rect getRect();
	bool isTracking();
};

