#pragma once
#include <random>
#include <time.h>
#include <climits>
#include "PridictionMatrix.h"

static class Randomizer
{
public:
	static int32_t randomInt(int32_t = 0, int32_t = INT_MAX);
	static int32_t* randomMatrix(uint32_t size, int32_t = 0, int32_t = INT_MAX);
	static int32_t* arrayRandomizer(Mat matrix);
};