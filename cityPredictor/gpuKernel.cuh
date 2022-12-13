
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void transposeMat(int*,int*,unsigned int);

__global__ void matFromProAtt(int*, int*, double*, double*, int);

int* launchGPUTranspose(int*, unsigned int);
int* launchCalculateMatFromProAtt(int* mat, double* pro, double* att, unsigned int size);