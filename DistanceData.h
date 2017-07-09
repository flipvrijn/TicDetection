#pragma once
#include <opencv2/imgproc/imgproc.hpp>
struct DistanceData {
	cv::Point pointOne;
	cv::Point pointTwo;
	float relativeTo;
};
