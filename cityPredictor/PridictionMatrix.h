#pragma once
#include <iostream>
using namespace std;

class Mat
{
private:
	int32_t* data;
	uint32_t size;

public:

	Mat();
	Mat(uint32_t size, int32_t defaultValue = 0);
//	~Mat();
	void displayMat();
	void clear();
	void reinit(uint32_t, int32_t defaultValue = 0);
	void setRandom(uint32_t,uint32_t);
	bool isEmpty();
	Mat transpose();
	uint32_t getSize();
	int64_t sum(int32_t);
	int32_t getElement(int32_t,int32_t);
	Mat calculateMatFromProductionAndAttraction(double*, double*);
};