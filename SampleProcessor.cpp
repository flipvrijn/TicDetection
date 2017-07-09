#include <vector>
#include <iostream>
#include <math.h>
#include <ctime>
#include <fstream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "boost/filesystem.hpp"

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "SampleProcessor.h"

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

SampleProcessor::SampleProcessor()
{
	this->faceDetection.setFile("haarcascade_frontalface_alt.xml");
	this->faceDetection.setSize(Size(100, 100));

	stringstream ss;
	ss << "out-" << time(NULL) << ".txt";
	this->setSaveFile(ss.str());
}


SampleProcessor::~SampleProcessor()
{
}

void SampleProcessor::addSamples(std::vector<std::pair<Label, boost::filesystem::path>> samples)
{
	this->samples = samples;
}

void SampleProcessor::setLoadFile(string loadFile)
{
	this->loadFile = loadFile;
}

void SampleProcessor::setSaveFile(string saveFile)
{
	this->saveFile = saveFile;
}

void SampleProcessor::process()
{
	// --- Perform feature extraction on each sample
	vector<Sample> sampleData;
	if (this->samples.size() > 0 && this->loadFile.empty())
	{
		for (size_t i = 0; i < this->samples.size(); i++)
		{
			cout << "Processed [" << i + 1 << "/" << this->samples.size() << "] samples" << endl;
			sampleData.push_back(this->calculateSample(this->samples[i], i + 1));
		}

		this->writeToFile(this->saveFile, sampleData);
		this->exportToFile(sampleData);
	}
	else if (!this->loadFile.empty() && this->samples.size() == 0)
	{
		sampleData = this->readFromFile(this->loadFile);
	}
	else
	{
		throw runtime_error("Error: No input source for the sample processor!");
	}

	if (sampleData.size() == 0)
	{
		throw runtime_error("Sample data is empty! Nothing to train with!");
	}

	stringstream stringOutput;

	vector<vector<Stat>> roundStats;
	vector<float> roundAverages;
	for (int r = 0; r < this->rounds; r++)
	{
		cout << "Running round [" << r + 1 << "/" << this->rounds << "]" << endl;
		
		// --- Create folds for cross-validation
		vector<vector<Sample>> folds = this->kFold(this->k, sampleData);

		vector<Stat> foldStats;
		Stat foldStat;
		// --- Run each fold
		for (size_t i = 0; i < folds.size(); i++)
		{
			stringOutput << endl << "Running fold [" << i + 1 << "/" << folds.size() << "]" << endl;

			// Test set
			vector<Sample> testSamples = folds[i];

			// Train set
			vector<Sample> trainSamples;
			vector<vector<Sample>> restOfFolds(folds);
			restOfFolds.erase(restOfFolds.begin() + i);
			for (size_t i = 0; i < restOfFolds.size(); i++)
			{
				vector<Sample> subSample = restOfFolds[i];
				trainSamples.insert(trainSamples.end(), subSample.begin(), subSample.end());
			}

			stringOutput << "Test size: " << testSamples.size() << ", train size: " << trainSamples.size() << endl
				<< "Constructing traindata" << endl;

			// --- Construct traindata
			vector<float> labels;
			vector<vector<float>> trainData;
			for (size_t i = 0; i < trainSamples.size(); i++)
			{
				Sample sample = trainSamples[i];
				for (size_t j = 0; j < sample.features.size(); j++)
				{
					labels.push_back(static_cast<float>(sample.label));
					trainData.push_back(sample.features[j]);
				}
			}

			stringOutput << "Transforming traindata" << endl;

			// --- Transform traindata into OpenCV representation
			Mat labelsMat(labels);
			Mat trainMat(trainData.size(), trainData[0].size(), CV_32FC1);
			for (int i = 0; i < trainMat.rows; i++)
			{
				for (int j = 0; j < trainMat.cols; j++)
				{
					trainMat.at<float>(i, j) = trainData[i][j];
				}
			}

			stringOutput << "Training the SVM" << endl;

			// --- Training the SVM
			CvSVMParams params;
			params.svm_type = CvSVM::C_SVC;
			params.kernel_type = CvSVM::RBF;
			params.gamma = 0.0001;
			params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);
			CvSVM svm;
			svm.train(trainMat, labelsMat, Mat(), Mat(), params);

			params = svm.get_params();
			//printf("D = %f, G = %f, C = %f, N = %f, P = %f\n", params.degree, params.gamma, params.coef0, params.nu, params.p);
			
			stringOutput << "Testing the SVM" << endl;

			// --- Testing the SVM
			labels.clear();
			vector<vector<float>> testData;
			for (size_t i = 0; i < testSamples.size(); i++)
			{
				Sample sample = testSamples[i];
				for (size_t j = 0; j < sample.features.size(); j++)
				{
					labels.push_back(static_cast<float>(sample.label));
					testData.push_back(sample.features[j]);
				}
			}

			int tp = 0, fp = 0, fn = 0, tn = 0;
			float testSampleSize = static_cast<float>(testSamples.size());
			for (int i = 0; i < testSampleSize; i++)
			{
				Mat features(testData[i]);
				float response = svm.predict(features);

				if (response == -1 && labels[i] == -1)
					tn++;
				if (response == 1 && labels[i] == 1)
					tp++;
				if (response == -1 && labels[i] == 1)
					fn++;
				if (response == 1 && labels[i] == -1)
					fp++;
			}

			// --- Store statistics
			foldStat.tp = (float)tp;
			foldStat.tn = (float)tn;
			foldStat.fp = (float)fp;
			foldStat.fn = (float)fn;

			foldStats.push_back(foldStat);
		}

		// --- Calculate round average
		vector<float> averages;
		for (size_t i = 0; i < foldStats.size(); i++)
		{
			averages.push_back(foldStats[i].performance());
		}
		roundStats.push_back(foldStats);

		float m = median(averages);
		roundAverages.push_back(m);
	}

	// --- Store statistics
	float m = median(roundAverages);
	stringstream ss;
	ss << "output/stats-" << time(NULL) << ".txt";
	printf("Median performance over %d rounds is %f.\nStats exported to %s.", this->rounds, m, ss.str());
	this->writeStats(ss.str(), roundStats);
}

SampleProcessor::Sample SampleProcessor::calculateSample(pair<Label, fs::path> s, int sampleId)
{
	VideoCapture capture;

	vector<vector<float>> dataSetVect;
	vector<vector<int>> typesVect;

	if (this->enableFeedback)
		namedWindow(this->windowName);

	// 1. Open the sample
	fs::path samplePath = fs::current_path() / s.second;
	string sampleFile = samplePath.string();
	capture.open(sampleFile);

	if (!capture.isOpened())
	{
		throw runtime_error("Error: Video input could not be opened!");
	}

	// 2. Set FPS information
	featureExtractor.setFPS(capture.get(CV_CAP_PROP_FPS));

	Mat frame;
	// 3. Loop over every frame
	while (true)
	{
		// 4. Read frame from input source
		capture >> frame;
		if (frame.empty())
			break;

		flip(frame, frame, 1);

		if (this->enableFeedback)
			featureExtractor.enableFeedback();

		if (faceDetection.detect(frame))
		{
			// 5. Pre-process each frame
			faceDetection.preprocess();
			
			// 6. Enable features
			for (size_t i = 0; i < this->enabledFeatures.size(); i++)
			{
				featureExtractor.enableFeature(this->enabledFeatures[i]);
			}
			featureExtractor.extractFeatures(faceDetection, s.first, sampleId);

			//cout << featureExtractor << endl;

			// 6. Construct data set & label for the SVM
			vector<float> values = featureExtractor.getValues();
			vector<int> types = featureExtractor.getTypes();
			if (values.size() > 0)
				dataSetVect.push_back(values);
			if (types.size() > 0)
				typesVect.push_back(types);
		}

		if (this->enableFeedback)
		{
			imshow(this->windowName, frame);
			if (waitKey(10) > 0) break;
		}
	}
		
	// 7. Reset the feature extractor for next use
	featureExtractor.reset();
	capture.release();
	destroyAllWindows();

	Sample sample;
	sample.sampleId = sampleId;
	sample.label = s.first;
	sample.features = dataSetVect;
	sample.types = typesVect;

	return sample;
}

void SampleProcessor::setKFolds(int kFold)
{
	this->k = kFold;
}

void SampleProcessor::setRounds(int rounds)
{
	this->rounds = rounds;
}

void SampleProcessor::setWindowSize(float windowSize)
{
	this->featureExtractor.setWindowSize(windowSize);
}

void SampleProcessor::setBlinkThreshold(float left, float right)
{
	this->featureExtractor.setBlinkThreshold(left, right);
}

vector<vector<SampleProcessor::Sample>> SampleProcessor::kFold(int k, vector<Sample> data)
{
	random_shuffle(data.begin(), data.end());

	vector<vector<Sample>> folds;

	// Split samples in k chunks
	int n = data.size();
	int at, pre = 0, i;
	for (pre = i = 0; i < k; i++)
	{
		at = (n + n*i) / k;
		// Grab subvector
		vector<Sample> chunk(data.begin() + pre, data.begin() + at);
		folds.push_back(chunk);
		pre = at;
	}

	return folds;
}

void SampleProcessor::exportToFile(vector<Sample> samples)
{
	ofstream fileOut;
	
	stringstream ss;
	ss << "output/export-" << time(NULL) << ".txt";
	fileOut.open(ss.str());

	// Print out header
	Sample header = samples[0];
	for (size_t i = 0; i < header.types[0].size(); i++)
	{
		fileOut << header.types[0][i] << " ";
	}
	fileOut << endl;

	// Print the samples vector
	for (size_t i = 0; i < samples.size(); i++)
	{
		Sample sample = samples[i];
		for (size_t j = 0; j < sample.features.size(); j++)
		{
			fileOut << sample.sampleId << " " << static_cast<int>(sample.label) << " ";
			for (size_t k = 0; k < sample.features[j].size(); k++)
			{
				fileOut << sample.features[j][k] << " ";
			}
			fileOut << endl;
		}
	}

	cout << "Sample data exported to " << ss.str() << endl;
}

void SampleProcessor::writeToFile(string filename, vector<Sample> samples)
{
	ofstream fileOut;
	fileOut.open(filename, ios::out | ios::trunc);
	if (!fileOut.is_open())
	{
		throw runtime_error("Error: Cannot export to file!");
	}

	boost::archive::text_oarchive oa(fileOut);

	oa << BOOST_SERIALIZATION_NVP(samples);

	fileOut.close();
}

vector<SampleProcessor::Sample> SampleProcessor::readFromFile(string filename)
{
	ifstream fileIn;
	fileIn.open(filename , ifstream::in);
	if (!fileIn.is_open())
	{
		throw runtime_error("Error: Cannot import from file!");
	}

	vector<Sample> samples;

	boost::archive::text_iarchive ia(fileIn);

	ia >> BOOST_SERIALIZATION_NVP(samples);

	fileIn.close();

	return samples;
}

void SampleProcessor::enableFeature(FeatureExtractor::FeatureType featureType)
{
	this->enabledFeatures.push_back(featureType);
}

void SampleProcessor::setFeedback(bool b)
{
	this->enableFeedback = b;
}

float SampleProcessor::median(vector<float> averages)
{
	float m;
	sort(averages.begin(), averages.end());

	if (averages.size() % 2 == 0)
		m = (averages[averages.size() / 2 - 1] + averages[averages.size() / 2]) / 2;
	else
		m = averages[averages.size() / 2];

	return m;
}

void SampleProcessor::writeStats(string filename, vector<vector<Stat>> stats)
{
	ofstream fileOut;
	fileOut.open(filename, ios::out | ios::trunc);
	if (!fileOut.is_open())
	{
		throw runtime_error("Error: Cannot export stats to file!");
	}

	fileOut << "Round Fold TP TN FP FN" << endl;

	for (size_t j = 0; j < stats.size(); j++)
	{
		for (size_t i = 0; i < stats[j].size(); i++)
		{
			fileOut << (j + 1) << " " << (i + 1) << " " << stats[j][i].tp << " " << stats[j][i].tn << " " << stats[j][i].fp << " " << stats[j][i].fn << endl;
		}
	}

	fileOut.close();
}