#include "PridictionMatrix.h"
#include "RandomGenerator.h"
#include "gpuKernel.cuh"

Mat::Mat()
{
	this->size = 0;
	this->data = NULL;
}

Mat::Mat(uint32_t size, int32_t defaultValue)
{
	this->data = new int32_t[size * size];
	this->size = size;

	memset(data,0,size * size * sizeof(int32_t));
}

void Mat::displayMat()
{
	for (int i = 0; i < this->size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			cout << data[i*size + j] << "\t";
		}
		cout << endl;
	}
}

void Mat::clear()
{
	if (data)
		delete(data);
	this->size = 0;
}

void Mat::reinit(uint32_t size, int32_t defaultValue)
{
	if (!this->isEmpty())
	{
		this->clear();
	}
	this->data = new int32_t [size * size];
	this->size = size;

	memset(this->data,defaultValue,size * size * sizeof(int));
}

void Mat::setRandom(uint32_t min, uint32_t max)
{
	int x = this->size;
	if (!this->isEmpty())
		this->clear();

	this->size = x;
	this->data = Randomizer::randomMatrix(x,min,max);
}

bool Mat::isEmpty()
{
	return this->size == 0;
}

Mat Mat::transpose()
{
	Mat x;
	x.size = this->size;

	x.data = launchGPUTranspose(this->data,this->size);

	return x;
}

uint32_t Mat::getSize()
{
	return this->size;
}

int64_t Mat::sum(int32_t row)
{
	int64_t x = 0;

	for (int i = row * this->size; i < row * this->size + this->size; i++)
	{
		x += this->data[i];
	}
	return x;
}

int32_t Mat::getElement(int32_t i, int32_t j)
{
	return this->data[i * this->size + j];
}

Mat Mat::calculateMatFromProductionAndAttraction(double* a, double* b)
{
	Mat x;

	x.size = this->size;

	x.data = launchCalculateMatFromProAtt(this->data, a, b, this->size);
	return x;
}
