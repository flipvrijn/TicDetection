#include <vector>
#include <iostream>
#include <fstream>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Feature.h"
#include "FeatureExtractor.h"
#include "SlidingWindow.h"
#include "PupilFeature.h"
#include "PupilData.h"
#include "BlinkFeature.h"
#include "DistanceFeature.h"
#include "DistanceData.h"
#include "MotionFeature.h"
#include "WrinkleFeature.h"

#include "CascadeDetection.h"
#include "PupilDetection.h"
#include "MotionDetection.h"

using namespace std;
using namespace cv;

FeatureExtractor::FeatureExtractor()
{
}

FeatureExtractor::~FeatureExtractor()
{
}

void FeatureExtractor::extractFeatures(FaceDetection faceDetection, Label label, int sampleId)
{
	// Get basic information
	this->faceDetection = faceDetection;
	this->currentLabel = label;
	this->currentSampleId = sampleId;

	// Reserve space
	this->features.resize(NUMTYPES);

	// Extract features
	if (this->featureEnabled(BLINKLEFT))
		this->extractBlinkLeft();
	if (this->featureEnabled(BLINKRIGHT))
		this->extractBlinkRight();
	if (this->featureEnabled(PUPILLEFT) && this->featureEnabled(PUPILRIGHT))
		this->extractPupils();
	if (this->featureEnabled(EYEMOTION) || this->featureEnabled(FACEMOTION))
		this->extractMotion();
	if (this->featureEnabled(MOUTHDISTANCE) || this->featureEnabled(MOUTHEYELEFTDISTANCE) || this->featureEnabled(MOUTHEYERIGHTDISTANCE))
		this->extractLandmarks();
	if (this->featureEnabled(WRINKLE))
		this->extractWrinkles();

	this->persist();
}

void FeatureExtractor::extractBlinkLeft()
{
	if (this->feedbackEnabled())
	{
		this->blinkDetection.enableFeedback();
		this->blinkDetection.setWindowName("Capture - Blinks");
	}

	if (!this->hasFeatureStatus(BLINKLEFT))
		this->setFeatureStatus(BLINKLEFT, false);

	Mat faceROI = this->faceDetection.getROI();

	this->blinkDetection.setFaceRectangle(this->faceDetection.getRect());
	this->blinkDetection.setFaceROI(faceROI);

	pair<Rect, Rect> eyeRegions = this->blinkDetection.getEyeRegions();

	BlinkData blinkLeft;
	BlinkFeature blinkLeftFeature(BLINKLEFT);

	// Detect blinks in left eye
	this->blinkDetection.setEyeTemplate(faceROI(eyeRegions.first));
	this->blinkDetection.setBlinkThreshold(this->blinkLeftThreshold);

	if (this->blinkDetection.detect())
		blinkLeft.blink = true;
	else
		blinkLeft.blink = false;
	this->blinkLeftSlidingWindow.storeData(blinkLeft);
	if (blinkLeftSlidingWindow.isFilled())
	{
		this->setFeatureStatus(BLINKLEFT, true);
		blinkLeftFeature.calculateValue(blinkLeftSlidingWindow.getData());
		this->storeFeature(blinkLeftFeature);
	}
}

void FeatureExtractor::extractBlinkRight()
{
	if (this->feedbackEnabled())
	{
		this->blinkDetection.enableFeedback();
		this->blinkDetection.setWindowName("Capture - Blinks");
	}

	if (!this->hasFeatureStatus(BLINKRIGHT))
		this->setFeatureStatus(BLINKRIGHT, false);

	Mat faceROI = this->faceDetection.getROI();

	this->blinkDetection.setFaceRectangle(this->faceDetection.getRect());
	this->blinkDetection.setFaceROI(faceROI);

	pair<Rect, Rect> eyeRegions = this->blinkDetection.getEyeRegions();

	BlinkData blinkRight;
	BlinkFeature blinkRightFeature(BLINKRIGHT);

	// Detect blinks in right eye
	this->blinkDetection.setEyeTemplate(faceROI(eyeRegions.second));
	this->blinkDetection.setBlinkThreshold(this->blinkRightThreshold);

	if (this->blinkDetection.detect())
		blinkRight.blink = true;
	else
		blinkRight.blink = false;
	this->blinkRightSlidingWindow.storeData(blinkRight);
	if (blinkRightSlidingWindow.isFilled())
	{
		this->setFeatureStatus(BLINKRIGHT, true);
		blinkRightFeature.calculateValue(blinkRightSlidingWindow.getData());
		this->storeFeature(blinkRightFeature);
	}
}

void FeatureExtractor::extractPupils()
{
	if (this->feedbackEnabled())
	{
		this->pupilDetection.enableFeedback();
		this->pupilDetection.setWindowName("Capture - Pupils");
	}
	// Set to not done
	if (!this->hasFeatureStatus(PUPILLEFT))
		this->setFeatureStatus(PUPILLEFT, false);
	if (!this->hasFeatureStatus(PUPILRIGHT))
		this->setFeatureStatus(PUPILRIGHT, false);

	Rect faceRect = this->faceDetection.getRect();
	Mat faceROI = this->faceDetection.getROI();

	this->pupilDetection.setFaceRectangle(faceRect);
	this->pupilDetection.setFaceROI(faceROI);
	if (!this->pupilDetection.detect())
	{
		throw runtime_error("Error: Pupil detection failed!");
	}
	vector<Eye> pupils = this->pupilDetection.getPupils();
	if (pupils.size() == 2)
	{
		PupilData pupilData;
		pupilData.leftPupil = pupils[0];
		pupilData.rightPupil = pupils[1];
		this->pupilSlidingWindow.storeData(pupilData);

		if (pupilSlidingWindow.isFilled())
		{
			// Change status
			this->setFeatureStatus(PUPILLEFT, true);
			this->setFeatureStatus(PUPILRIGHT, true);

			PupilFeature leftPupilFeature(PUPILLEFT), rightPupilFeature(PUPILRIGHT);
			vector<PupilData> pupilData = pupilSlidingWindow.getData();

			leftPupilFeature.calculateValue(pupilData, 0);
			rightPupilFeature.calculateValue(pupilData, 1);
			this->storeFeature(leftPupilFeature);
			this->storeFeature(rightPupilFeature);
		}
	}
}

void FeatureExtractor::extractLandmarks()
{
	if (!this->hasFeatureStatus(MOUTHDISTANCE))
		this->setFeatureStatus(MOUTHDISTANCE, false);
	if (!this->hasFeatureStatus(MOUTHEYELEFTDISTANCE))
		this->setFeatureStatus(MOUTHEYELEFTDISTANCE, false);
	if (!this->hasFeatureStatus(MOUTHEYERIGHTDISTANCE))
		this->setFeatureStatus(MOUTHEYERIGHTDISTANCE, false);

	Mat matImg = this->faceDetection.getImage();
	Rect faceRect(this->faceDetection.getRect());
	Mat faceROI = this->faceDetection.getROI();

	vector<Point> landmarks = this->faceDetection.getLandmarks();

	vector<Scalar> scalars = {
		Scalar(0, 0, 255), // red -> right corner left eye, 2 & 3
		Scalar(0, 255, 0), // green -> left corner right eye, 4 & 5
		Scalar(255, 0, 0), // blue -> left mouth corner, 6 & 7
		Scalar(0, 255, 255), // yellow -> right mouth corner, 8 & 9
		Scalar(255, 0, 255), // purple -> left corner left eye, 10 & 11
		Scalar(255, 255, 0), // cyan -> right corner right eye, 12 & 13
		Scalar(255, 255, 255) // white -> nose, 14 & 15
	};
	int colorIndex = 0;
	for (size_t i = 0; i < landmarks.size(); i++)
	{
		circle(matImg, landmarks[i], 3, scalars[colorIndex], CV_FILLED);
		colorIndex++;
	}

	this->setFeatureStatus(MOUTHDISTANCE, true);
	this->setFeatureStatus(MOUTHEYELEFTDISTANCE, true);
	this->setFeatureStatus(MOUTHEYERIGHTDISTANCE, true);

	// Distance features
	// Mouth points
	DistanceFeature mouthDistance(DistanceFeature::EUCLEDIAN, MOUTHDISTANCE);
	DistanceData mouthData = {
		landmarks[2],
		landmarks[3],
		faceROI.cols
	};
	mouthDistance.calculateValue(mouthData);
	this->storeFeature(mouthDistance);
	// LEFT Mouth and LEFT eye 
	DistanceFeature mouthEyeLeftDistance(DistanceFeature::EUCLEDIAN, MOUTHEYELEFTDISTANCE);
	DistanceData mouthEyeLeftData = {
		landmarks[2],
		landmarks[4],
		faceROI.rows
	};
	mouthEyeLeftDistance.calculateValue(mouthEyeLeftData);
	this->storeFeature(mouthEyeLeftDistance);
	// RIGHT Mouth and RIGHT eye 
	DistanceFeature mouthEyeRightDistance(DistanceFeature::EUCLEDIAN, MOUTHEYERIGHTDISTANCE);
	DistanceData mouthEyeRightData = {
		landmarks[3],
		landmarks[5],
		faceROI.rows
	};
	mouthEyeRightDistance.calculateValue(mouthEyeRightData);
	this->storeFeature(mouthEyeRightDistance);
}

void FeatureExtractor::extractMotion()
{
	if (this->feedbackEnabled())
	{
		this->motionDetection.enableFeedback();
		this->motionDetection.setWindowName("Capture - Motion");
	}

	// Set to not done
	if (!this->hasFeatureStatus(EYEMOTION))
		this->setFeatureStatus(EYEMOTION, false);
	if (!this->hasFeatureStatus(FACEMOTION))
		this->setFeatureStatus(FACEMOTION, false);

	Rect faceRect = this->faceDetection.getRect();
	Mat faceROI = this->faceDetection.getROI();
	// Dependency
	this->pupilDetection.setFaceRectangle(faceRect);
	this->pupilDetection.setFaceROI(faceROI);
	if (!this->pupilDetection.detect())
	{
		throw runtime_error("Error: Pupil detection failed!");
	}

	this->motionDetection.setRect(faceRect);
	vector<Rect> eyeRegions = this->pupilDetection.getEyeRegions();
	Rect eyeRect = Rect(eyeRegions[0].x, eyeRegions[0].y, eyeRegions[0].width + eyeRegions[1].width, eyeRegions[0].height);
	this->motionDetection.setEyeRect(eyeRect);
	this->motionDetection.feedFrame(this->faceDetection.getImage());

	if (this->motionDetection.isFilled())
	{
		if (this->motionDetection.detect())
		{
			// Motion pixels within eye region
			vector<Rect> eyeRegions = this->pupilDetection.getEyeRegions();
			Mat diff = this->motionDetection.getDiff(this->faceDetection.getRect());
			Mat diffLeftEyeRegion = diff(eyeRegions[0]);
			Mat diffRightEyeRegion = diff(eyeRegions[1]);
			float totalPixelsEyes = static_cast<float>((diffLeftEyeRegion.rows * diffLeftEyeRegion.cols) + (diffRightEyeRegion.rows * diffRightEyeRegion.cols));
			float whitePixelsEyes = static_cast<float>(countNonZero(diffLeftEyeRegion) + countNonZero(diffRightEyeRegion));

			this->eyeMotionSlidingWindow.storeData((whitePixelsEyes / totalPixelsEyes));

			// Motion pixels everywhere except eye region and mouth region
			Mat img = this->faceDetection.getImage().clone();
			vector<Point> landmarks = this->faceDetection.getLandmarks();
			Point leftMouthCorner = landmarks[2], rightMouthCorner = landmarks[3];
			Point nose = landmarks[6];
			Rect mouthRegion(leftMouthCorner.x, nose.y, rightMouthCorner.x - leftMouthCorner.x, (leftMouthCorner.y - nose.y) * 2);
			Mat diffMouth = this->motionDetection.getDiff(mouthRegion);
			
			float totalPixelsFace = static_cast<float>(diff.rows * diff.cols);
			float whitePixelsFace = static_cast<float>(countNonZero(diff));
			float whitePixelsMouth = static_cast<float>(countNonZero(diffMouth));

			this->faceMotionSlidingWindow.storeData((whitePixelsFace - whitePixelsEyes - whitePixelsMouth) / totalPixelsFace);
		}
	}

	if (this->eyeMotionSlidingWindow.isFilled())
	{
		// Change status
		this->setFeatureStatus(EYEMOTION, true);

		MotionFeature motionFeature(EYEMOTION);
		motionFeature.calculateValue(this->eyeMotionSlidingWindow.getData());
		this->storeFeature(motionFeature);
	}

	if (this->faceMotionSlidingWindow.isFilled())
	{
		// Change status
		this->setFeatureStatus(FACEMOTION, true);

		MotionFeature motionFeature(FACEMOTION);
		motionFeature.calculateValue(this->faceMotionSlidingWindow.getData());
		this->storeFeature(motionFeature);
	}
}

void FeatureExtractor::extractWrinkles()
{
	if (this->feedbackEnabled())
	{
		this->wrinkleDetection.enableFeedback();
		this->wrinkleDetection.setWindowName("Capture - Wrinkles");
	}

	// Dependency
	this->pupilDetection.setFaceRectangle(this->faceDetection.getRect());
	this->pupilDetection.setFaceROI(this->faceDetection.getROI());
	if (!this->pupilDetection.detect())
	{
		throw runtime_error("Error: Pupil detection failed!");
	}
	vector<Rect> eyeRegions = this->pupilDetection.getEyeRegions();
	this->wrinkleDetection.setDetectionRegion(this->faceDetection.getROI(), Rect(eyeRegions[0].x, eyeRegions[0].y, eyeRegions[0].width + eyeRegions[1].width, eyeRegions[0].height));

	if (this->wrinkleDetection.detect())
	{
		Mat detectedEdges = this->wrinkleDetection.getDetectedEdges();
		vector<Rect> eyeRegions = this->pupilDetection.getEyeRegions();
		
		float numNonZero = static_cast<float>(countNonZero(detectedEdges));
		float totalPixels = static_cast<float>(detectedEdges.rows * detectedEdges.cols);

		WrinkleFeature wrinkleFeature(WRINKLE);
		wrinkleFeature.setValue(numNonZero / totalPixels);
		this->storeFeature(wrinkleFeature);
	}
}

vector<float> FeatureExtractor::getValues()
{
	vector<float> values;
	if (this->allFeaturesDone())
	{
		for (size_t i = 0; i < this->features.size(); i++)
		{
			if (!this->features[i].isEmpty())
			{
				values.push_back(this->features[i].getValue());
			}
		}
	}

	return values;
}

vector<int> FeatureExtractor::getTypes()
{
	vector<int> types;
	if (this->allFeaturesDone())
	{
		for (size_t i = 0; i < this->features.size(); i++)
		{
			if (!this->features[i].isEmpty())
			{
				types.push_back(this->features[i].getType());
			}
		}
	}

	return types;
}

vector<vector<float>> FeatureExtractor::getAllValues()
{
	vector<vector<float>> values;
	for (size_t i = 0; i < this->history.size(); i++)
	{
		FeatureVector row = this->history[i];
		vector<float> features;
		for (size_t j = 0; j < row.features.size(); j++)
		{
			Feature feature = row.features[j];
			if (!feature.isEmpty())
			{
				features.push_back(feature.getValue());
			}
		}
		values.push_back(features);
	}

	return values;
}

vector<float> FeatureExtractor::getLabels()
{
	vector<float> labels;
	for (size_t i = 0; i < this->history.size(); i++)
	{
		FeatureVector row = this->history[i];
		labels.push_back(static_cast<float>(row.label));
	}

	return labels;
}

vector<Feature> FeatureExtractor::getFeatureVector()
{
	return this->features;
}

void FeatureExtractor::setFPS(double fps)
{
	if (this->fps != fps)
	{
		this->fps = fps;

		size_t windowSize = (size_t)(this->windowSize * this->fps);

		this->pupilSlidingWindow.setSize(windowSize);
		this->blinkLeftSlidingWindow.setSize(windowSize);
		this->blinkRightSlidingWindow.setSize(windowSize);
		this->eyeMotionSlidingWindow.setSize(windowSize);
		this->faceMotionSlidingWindow.setSize(windowSize);
	}
}

void FeatureExtractor::setWindowSize(float windowSize)
{
	this->windowSize = windowSize;
}

void FeatureExtractor::setBlinkThreshold(float left, float right)
{
	this->blinkLeftThreshold = left;
	this->blinkRightThreshold = right;
}

void FeatureExtractor::enableFeedback()
{
	this->feedback = true;
}

bool FeatureExtractor::feedbackEnabled()
{
	return this->feedback;
}

void FeatureExtractor::enableFeature(FeatureType featureType)
{
	if (!this->featureEnabled(featureType))
	{
		this->enabledFeatures.push_back(featureType);
	}
}

bool FeatureExtractor::featureEnabled(FeatureType featureType)
{
	return (find(this->enabledFeatures.begin(), this->enabledFeatures.end(), featureType) != this->enabledFeatures.end());
}

void FeatureExtractor::storeFeature(Feature feature)
{
	if (feature.getType() < this->features.size())
	{
		this->features[feature.getType()] = feature;
	}
}

void FeatureExtractor::persist()
{
	if (this->allFeaturesDone())
	{
		FeatureVector featureVector;
		featureVector.label = this->currentLabel;
		featureVector.sampleID = this->currentSampleId;
		featureVector.features = this->features;
		this->history.push_back(featureVector);
	}
}

ostream& operator<<(ostream& strm, FeatureExtractor& fe)
{
	strm << "[";
	vector<float> values = fe.getValues();
	for (size_t i = 0; i < values.size(); i++)
	{
		strm << static_cast<float>(values[i]) << ",";
	}
	strm << "]";
	return strm;
}

void FeatureExtractor::setFeatureStatus(FeatureType type, bool status)
{
	for (size_t i = 0; i < this->featureStatus.size(); i++)
	{
		pair<bool, FeatureType> currentStatus = featureStatus[i];
		if (currentStatus.second == type)
		{
			currentStatus.first = status;
			featureStatus[i] = currentStatus;
			return;
		}
	}

	pair<bool, FeatureType> s(status, type);
	this->featureStatus.push_back(s);
}

bool FeatureExtractor::allFeaturesDone()
{
	for (size_t i = 0; i < this->featureStatus.size(); i++)
	{
		pair<bool, FeatureType> currentStatus = featureStatus[i];
		if (!currentStatus.first)
			return false;
	}

	return true;
}

bool FeatureExtractor::hasFeatureStatus(FeatureType type)
{
	for (size_t i = 0; i < this->featureStatus.size(); i++)
	{
		pair<bool, FeatureType> currentStatus = featureStatus[i];
		if (currentStatus.second == type)
			return true;
	}

	return false;
}

void FeatureExtractor::reset()
{
	this->pupilSlidingWindow.reset();
	this->blinkLeftSlidingWindow.reset();
	this->blinkRightSlidingWindow.reset();
	this->eyeMotionSlidingWindow.reset();
	this->faceMotionSlidingWindow.reset();

	this->blinkDetection.reset();
	this->motionDetection.reset();

	this->featureStatus.clear();
	this->features.clear();

}