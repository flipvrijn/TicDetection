#pragma once
#include "boost/filesystem.hpp"
#include "Label.h"

class DataLoader
{
private:
	std::vector<std::pair<Label, boost::filesystem::path>> samples;
	std::vector<std::vector<std::pair<Label, boost::filesystem::path>>> folds;
	bool validFile(boost::filesystem::path);
public:
	DataLoader();
	~DataLoader();
	void loadData(std::string root);
	std::vector<std::pair<Label, boost::filesystem::path>> getSamples();
	size_t size();
};

