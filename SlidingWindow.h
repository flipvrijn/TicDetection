#pragma once
#include <vector>

using namespace std;

template <class T> 
class SlidingWindow
{
private:
	std::vector<T> data;
	size_t vectorSize;
	bool filled;
public:
	SlidingWindow();
	SlidingWindow(size_t size);
	~SlidingWindow();
	bool isFilled();
	std::vector<T> getData();
	void storeData(T data);
	void setSize(size_t size);
	size_t size();
	void reset();
};

template <class T>
SlidingWindow<T>::SlidingWindow()
{

}

template <class T>
SlidingWindow<T>::SlidingWindow(size_t size)
{
	this->data.reserve(size);
}

template <class T>
SlidingWindow<T>::~SlidingWindow()
{

}

template <class T>
void SlidingWindow<T>::setSize(size_t size)
{
	this->vectorSize = size;
}

template <class T>
bool SlidingWindow<T>::isFilled()
{
	return this->filled;
}

template <class T>
vector<T> SlidingWindow<T>::getData()
{
	vector<T> data = this->data;
	return data;
}

template <class T>
void SlidingWindow<T>::storeData(T data)
{
	if (this->data.size() == this->vectorSize)
	{
		this->filled = true;
		this->data.erase(this->data.begin()); // remove oldest sample
	}

	this->data.push_back(data);
}

template <class T>
size_t SlidingWindow<T>::size()
{
	return this->data.size();
}

template <class T>
void SlidingWindow<T>::reset()
{
	this->data.clear();
	this->filled = false;
}