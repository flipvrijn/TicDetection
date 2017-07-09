#include <iostream>

//------- OpenCV -------
#include <opencv2/highgui/highgui.hpp>

//------- Data loading -------
#include "DataLoader.h"

//------- Sample processor -------
#include "SampleProcessor.h"

//------- Boost -------
#include "boost/program_options.hpp"
#include "boost/foreach.hpp"
#include "boost/tokenizer.hpp"

using namespace std;
namespace po = boost::program_options;

string listAvailableFeatures(map<string, FeatureExtractor::FeatureType> featureMap)
{
	stringstream ss;
	ss << "Available features: ";
	for (map<string, FeatureExtractor::FeatureType>::iterator it = featureMap.begin(); it != featureMap.end(); ++it)
	{
		ss << it->first << " ";
	}
	ss << "all";

	return ss.str();
}

int main(int argc, char **argv)
{
	try
	{
		string inputDir = "", enabledFeaturesString = "", loadFile = "", saveFile = "";
		int kFold = 10, rounds = 200;
		bool enableFeedback = false;

		float windowSize = 0.8;
		float blinkLeftThreshold = 0.85, blinkRightThreshold = 0.85;

		map<string, FeatureExtractor::FeatureType> featureMap;
		featureMap["eyemotion"] = FeatureExtractor::FeatureType::EYEMOTION;
		featureMap["facemotion"] = FeatureExtractor::FeatureType::FACEMOTION;
		featureMap["wrinkle"] = FeatureExtractor::FeatureType::WRINKLE;
		featureMap["mouthdistance"] = FeatureExtractor::FeatureType::MOUTHDISTANCE;
		featureMap["moutheyeleftdistance"] = FeatureExtractor::FeatureType::MOUTHEYELEFTDISTANCE;
		featureMap["moutheyerightdistance"] = FeatureExtractor::FeatureType::MOUTHEYERIGHTDISTANCE;
		featureMap["blinkleft"] = FeatureExtractor::FeatureType::BLINKLEFT;
		featureMap["blinkright"] = FeatureExtractor::FeatureType::BLINKRIGHT;
		featureMap["pupilleft"] = FeatureExtractor::FeatureType::PUPILLEFT;
		featureMap["pupilright"] = FeatureExtractor::FeatureType::PUPILRIGHT;

		// 1. Parse command line
		po::options_description desc("Usage");
		desc.add_options()
			("help,h", "Produces help message")
			("dir,d", po::value<string>(&inputDir), "Data input directory")
			("load,l", po::value<string>(&loadFile), "Load features dump file")
			("save,s", po::value<string>(&saveFile), "Save features to dump file (default: out-<time>.txt)")
			("feedback,F", "Enable feedback")
			("kfold,k", po::value<int>(&kFold), "Set number of k-folds (default: 10)")
			("features,f", po::value<string>(&enabledFeaturesString), "Set features to detect")
			("list-features,L", "List all features possible to detect")
			("rounds,r", po::value<int>(&rounds), "Set number of iterations folds are executed (default: 200)")
			("win-size,w", po::value<float>(&windowSize), "Set sliding window size (default: 0.8)")
			("bl-thresh", po::value<float>(&blinkLeftThreshold), "Set blink left threshold (default: 0.85)")
			("br-thresh", po::value<float>(&blinkRightThreshold), "Set blink right threshold (default: 0.85)");
			;

		po::variables_map vm;
		try
		{
			po::store(po::parse_command_line(argc, argv, desc), vm);

			if (vm.count("help"))
			{
				cout << desc << endl;
				return 1;
			}

			po::notify(vm);

			if (vm.count("feedback"))
			{
				enableFeedback = true;
			}

			if (vm.count("list-features"))
			{
				cout << listAvailableFeatures(featureMap);
				return 1;
			}
		}
		catch (po::error &e)
		{
			stringstream ss;
			ss << "Error: " << e.what();
			throw runtime_error(ss.str());
		}

		if (inputDir.empty() && loadFile.empty())
			throw runtime_error("Error: The data input directory name or dump filename cannot be empty!");

		if (enabledFeaturesString.empty() && loadFile.empty())
			throw runtime_error("Error: At least one feature must be enabled!");

		if (windowSize < 0 || windowSize > 1)
			throw runtime_error("Error: The sliding window size must be a value from 0.0 - 1.0");

		if (blinkLeftThreshold < 0 || blinkLeftThreshold > 1 || blinkRightThreshold < 0 || blinkRightThreshold > 1)
			throw runtime_error("Error: The blink threshold must be a value from 0.0 - 1.0");

		// 2. Parse string of features to enable
		boost::char_separator<char> sep(",");
		boost::tokenizer<boost::char_separator<char>> tokens(enabledFeaturesString, sep);
		vector<FeatureExtractor::FeatureType> enabledFeatures;
		BOOST_FOREACH(const string& t, tokens)
		{
			if (t == "all")
			{
				for (map<string, FeatureExtractor::FeatureType>::iterator it = featureMap.begin(); it != featureMap.end(); ++it)
				{
					cout << it->first << " ";
					enabledFeatures.push_back(it->second);
				}
			}
			else
			{
				if (featureMap.find(t) == featureMap.end())
				{
					stringstream ss;
					ss << "Error: Feature '" << t << "' does not exist." << endl;
					ss << listAvailableFeatures(featureMap);
					throw runtime_error(ss.str());
				}
				enabledFeatures.push_back(featureMap[t]);
			}
		}

		// 3. Load input data if needed
		DataLoader dataLoader;
		SampleProcessor processor;
		if (!inputDir.empty())
		{
			dataLoader.loadData(inputDir);
			processor.addSamples(dataLoader.getSamples());

			cout << "Loaded " << dataLoader.size() << " samples" << endl;
		}

		if (!loadFile.empty())
			processor.setLoadFile(loadFile);

		if (!saveFile.empty())
			processor.setSaveFile(saveFile);

		// 4. Process samples
		processor.setFeedback(enableFeedback);
		processor.setKFolds(kFold);
		processor.setRounds(rounds);
		processor.setWindowSize(windowSize);
		processor.setBlinkThreshold(blinkLeftThreshold, blinkRightThreshold);
		for (size_t i = 0; i < enabledFeatures.size(); i++)
		{
			processor.enableFeature(enabledFeatures[i]);
		}
		processor.process();

		cv::destroyAllWindows();
	}
	catch (exception& e)
	{
		cerr << e.what() << endl;
		cin.get();
		return -1;
	}

	return 0;
}