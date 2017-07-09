#pragma once
class CascadeDetection
{
protected:
	cv::Rect rect;
	cv::Mat ROI;
	cv::Mat image;
	cv::Size size;
	cv::CascadeClassifier cascadeClassifier;
public:
	CascadeDetection();
	CascadeDetection(cv::String cascadeFile);
	CascadeDetection(cv::String cascadeFile, cv::Size size);
	~CascadeDetection();
	bool detect(cv::Mat frame);
	void setSize(cv::Size size);
	void setFile(cv::String);
	cv::Rect getRect();
	cv::Mat getROI();
	cv::Mat getImage();
};