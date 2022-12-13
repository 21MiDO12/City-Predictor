#include <iostream>
#include <vector>
#include "PridictionMatrix.h"
#include "RandomGenerator.h"

using namespace std;

double* evaluate(int32_t* results, double* cos, Mat mat, uint32_t size)
{
	double x = 0, * res = new double[size];
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			x += cos[j] * mat.getElement(i,j);
		}
		x = ((float)results[i]) / x;
		res[i] = x;
		x = 0;
	}

	return res;
}

int main(int argc, char** argv)
{
	uint32_t n;
	int32_t* production, * attraction;
	double* a, * b;
	Mat* x , y;
	bool flib = true;

	cout << "Enter your space size : ";
	cin >> n;

	if (n > 1024)
		n = 1024;

	x = new Mat(n);
//	x->displayMat();
	x->setRandom(100,1000);
	x->displayMat();
	
	production = Randomizer::arrayRandomizer(*x);
	attraction = Randomizer::arrayRandomizer(x->transpose());

	cout << endl << endl;

	cout << "Production : ";
	for (int i = 0; i < n; i++)
	{
		cout << production[i] << "\t";
	}
	cout << "\nAttraction : ";
	for (int i = 0; i < n; i++)
	{
		cout << attraction[i] << "\t";
	}
	cout << endl << endl;

	b = new double[n];
	
	for (int i = 0; i < n; i++)
	{
		b[i] = 1;
	}

	for (int64_t i = 0; i < 1000000; i++)
	{
		if (flib)
		{
			a = evaluate(production, b, *x, n);
		}
		else
		{
			b = evaluate(attraction, a, *x, n);
		}
	}

	cout << "\n\n------------------------\n\n";

	cout << "Production Values: ";
	for (int i = 0; i < n; i++)
	{
		cout << a[i] << "\t";
	}
	cout << "\nAttraction Values : ";
	for (int i = 0; i < n; i++)
	{
		cout << b[i] << "\t";
	}
	cout << endl << endl;

	y = x->calculateMatFromProductionAndAttraction(a, b);
	y.displayMat();

	return 0;
}