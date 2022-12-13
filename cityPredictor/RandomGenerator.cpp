#include "RandomGenerator.h"

int32_t Randomizer::randomInt(int32_t min, int32_t max)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<std::mt19937::result_type> dist(min,max);
    return dist(rng);
}

int32_t* Randomizer::randomMatrix(uint32_t size, int32_t min, int32_t max)
{
    if (size <= 0)
        return NULL;

    int32_t* x = new int32_t [size * size];

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            x[i * size + j] = Randomizer::randomInt(min,max);
        }
    }

    return x;
}

int32_t* Randomizer::arrayRandomizer(Mat matrix)
{
    int32_t s = matrix.getSize();
    int32_t* x = new int32_t[s];

    for (int i = 0; i < s; i++)
    {
        x[i] = matrix.sum(i) + Randomizer::randomInt(0,1000);
    }

    return x;
}
