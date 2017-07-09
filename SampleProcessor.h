#pragma once
#include "FaceDetection.h"
#include "FeatureExtractor.h"
#include "Label.h"
class SampleProcessor
{
private:
	std::vector<float> performance;
	std::vector<FeatureExtractor::FeatureType> enabledFeatures;
	std::vector<std::pair<Label, boost::filesystem::path>> samples;
	bool enableFeedback = false;
	cv::String windowName = "Capture Tics";
	std::string loadFile, saveFile;
	int k, rounds;

	FaceDetection faceDetection;
	FeatureExtractor featureExtractor;
public:
	struct Sample {
		int sampleId;
		Label label;
		std::vector<std::vector<float>> features;
		std::vector<std::vector<int>> types;

		template <typename Archive>
		void serialize(Archive& ar, const unsigned int version)
		{
			ar & sampleId;
			ar & label;
			ar & features;
			ar & types;
		}
	};

	struct Stat {
		float tp, tn, fp, fn;

		float performance()
		{
			return (tp + tn) / (tp + tn + fp + fn);
		}
	};

	SampleProcessor();
	~SampleProcessor();

	void enableFeature(FeatureExtractor::FeatureType);
	void setFeedback(bool);
	void setKFolds(int);
	void setRounds(int);
	void setWindowSize(float);
	void setBlinkThreshold(float, float);
	void addSamples(std::vector<std::pair<Label, boost::filesystem::path>>);
	void setLoadFile(std::string);
	void setSaveFile(std::string);
	void process();
private:
	std::vector<std::vector<Sample>> kFold(int, std::vector<Sample>);
	void exportToFile(std::vector<Sample>);
	void writeToFile(std::string, std::vector<Sample>);
	std::vector<Sample> readFromFile(std::string);
	Sample calculateSample(std::pair<Label, boost::filesystem::path>, int);
	float median(vector<float>);
	void writeStats(string, vector<vector<Stat>>);
};

