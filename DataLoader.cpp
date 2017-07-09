#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/filesystem.hpp"
#include <iostream>
#include "DataLoader.h"

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

DataLoader::DataLoader()
{
}


DataLoader::~DataLoader()
{
}


void DataLoader::loadData(string root)
{
	// 1. Make sure the input directory structure is correct
	fs::path basePath(root);
	if (!fs::exists(basePath))
	{
		stringstream ss;
		ss << "Error: Base path '" << root << "' does not exist!";
		throw runtime_error(ss.str());
	}

	fs::path posPath(basePath);
	posPath /= "pos";
	fs::path negPath(basePath);
	negPath /= "neg";

	if (!fs::exists(posPath) || !fs::exists(negPath) || !fs::is_directory(negPath) || !fs::is_directory(posPath))
	{
		stringstream ss;
		ss << "Error: The directory 'pos' or 'neg' does not exist in " << root;
		throw runtime_error(ss.str());
	}

	// 2. Recursively iterate over all files inside the pos and neg paths
	fs::recursive_directory_iterator itrPos(posPath);
	while (itrPos != fs::recursive_directory_iterator())
	{
		fs::path file = itrPos->path().string();
		if (this->validFile(file))
		{
			pair<Label, fs::path> sample(Label::TIC, file);
			this->samples.push_back(sample);
		}
		++itrPos;
	}
	
	fs::recursive_directory_iterator itrNeg(negPath);
	while (itrNeg != fs::recursive_directory_iterator())
	{
		fs::path file(itrNeg->path().string());
		if (this->validFile(file))
		{
			pair<Label, fs::path> sample(Label::NOTIC, file);
			this->samples.push_back(sample);
		}
		++itrNeg;
	}
}

vector<pair<Label, fs::path>> DataLoader::getSamples()
{
	return this->samples;
}

size_t DataLoader::size()
{
	return this->samples.size();
}

bool DataLoader::validFile(fs::path file)
{
	return (fs::is_regular_file(file) && file.extension() == ".mov");
}