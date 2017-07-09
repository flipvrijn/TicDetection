#pragma once

#include "Feature.h"
#include "SlidingWindow.h"

#include "BlinkDetection.h"
#include "WrinkleDetection.h"
#include "MotionDetection.h"
#include "FaceDetection.h"
#include "CascadeDetection.h"
#include "PupilDetection.h"
#include "PupilFeature.h"
#include "PupilData.h"
#include "BlinkData.h"
#include "Label.h"

struct FeatureVector{
	int sampleID;
	Label label;
	std::vector<Feature> features;
};

class FeatureExtractor
{
private:
	std::vector<FeatureVector> history;
	std::vector<Feature> features;
	SlidingWindow<PupilData> pupilSlidingWindow;
	SlidingWindow<BlinkData> blinkLeftSlidingWindow, blinkRightSlidingWindow;
	SlidingWindow<float> eyeMotionSlidingWindow, faceMotionSlidingWindow;
	Label currentLabel;
	int currentSampleId;
	double fps = 0.0;
	float windowSize;
	bool feedback = false;
	float blinkLeftThreshold, blinkRightThreshold;

	// Detector instances
	FaceDetection faceDetection;
	PupilDetection pupilDetection;
	MotionDetection motionDetection;
	WrinkleDetection wrinkleDetection;
	BlinkDetection blinkDetection;
public:
	enum FeatureType {
		PUPILLEFT,
		PUPILRIGHT,
		BLINKLEFT,
		BLINKRIGHT,
		MOUTHDISTANCE,
		MOUTHEYELEFTDISTANCE,
		MOUTHEYERIGHTDISTANCE,
		EYEMOTION,
		FACEMOTION,
		WRINKLE,
		NUMTYPES
	};

	FeatureExtractor();
	~FeatureExtractor();
	
	std::vector<int> getTypes();
	std::vector<float> getValues();
	std::vector<std::vector<float>> getAllValues();
	std::vector<float> getLabels();
	std::vector<Feature> getFeatureVector();
	void extractFeatures(FaceDetection, Label, int);

	void setFPS(double);
	void setWindowSize(float);
	void setBlinkThreshold(float, float);

	void enableFeedback();
	bool feedbackEnabled();
	void enableFeature(FeatureType);
	void reset();

	friend std::ostream& operator<<(std::ostream&, FeatureExtractor&);
private:
	std::vector<FeatureType> enabledFeatures;
	std::vector<std::pair<bool, FeatureType>> featureStatus;

	void extractBlinkLeft();
	void extractBlinkRight();
	void extractPupils();
	void extractLandmarks();
	void extractMotion();
	void extractWrinkles();

	bool featureEnabled(FeatureType);

	bool allFeaturesDone();
	bool hasFeatureStatus(FeatureType);
	void setFeatureStatus(FeatureType, bool);

	void persist();
	void storeFeature(Feature);
};

