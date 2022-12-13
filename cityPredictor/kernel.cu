#include <stdio.h>
#include <cmath>
#include <corecrt_malloc.h>
#include "gpuKernel.cuh"

__global__ void transposeMat(int* a, int* b ,unsigned int size)
{
    int i = blockIdx.x * size + threadIdx.x;
    int j = threadIdx.x * size + blockIdx.x;

    b[j] = a[i];
}

__global__ void matFromProAtt(int* mat , int* res, double* pro, double* att, int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    res[i * size + j] = lround((double)mat[i * size + j] * pro[i] * att[j]);

//    printf("I am %d and my result id %d\n", i * size + j, res[i * size + j]);
}

int* launchGPUTranspose(int* a, unsigned int size)
{
    int* dev_a = 0;
    int* dev_res = 0;
    int* res = a;

    cudaError_t status;

    status = cudaSetDevice(0);
    if (status != cudaSuccess)
    {
        fprintf(stderr,"Can't Find GPU");
        return NULL;
    }

    status = cudaMalloc((void**)&dev_a,size * size *sizeof(int));
    status = cudaMalloc((void**)&dev_res, size * size * sizeof(int));

    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Allocate Memory");
        goto FreeData;
    }

    status = cudaMemcpy(dev_a,a,size * size * sizeof(int),cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Allocate Memory");
        goto FreeData;
    }

    status = cudaMemcpy(dev_a,a,sizeof(int) * size * size, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Copy Memory");
        goto FreeData;
    }

    transposeMat << <size, size >> > (dev_a,dev_res,size);

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Kernel didn't launch correctly");
        goto FreeData;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Error happened in a block");
        goto FreeData;
    }

    status = cudaMemcpy(res, dev_res, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Copy Memory From GPU");
        goto FreeData;
    }

FreeData:
    
    cudaFree(dev_res);
    cudaFree(dev_a);

    status = cudaDeviceReset();
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Reset Cuda");
    }

    return res;
}

int* launchCalculateMatFromProAtt(int* mat, double* pro, double* att,unsigned int size)
{
    int* res = mat;
    int* dev_mat = 0, * dev_res = 0;
    double* dev_pro = 0, * dev_att = 0;

    cudaError_t status;

    status = cudaSetDevice(0);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Find GPU");
        goto FreeData;
    }

    status = cudaMalloc((void**)&dev_mat, size * size * sizeof(int));
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Allocate Memory on GPU");
        goto FreeData;
    }

    status = cudaMalloc((void**)&dev_res, size * size * sizeof(int));
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Allocate Memory on GPU");
        goto FreeData;
    }

    status = cudaMalloc((void**)&dev_pro, size * sizeof(double));
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Allocate Memory on GPU");
        goto FreeData;
    }

    status = cudaMalloc((void**)&dev_att, size * sizeof(double));
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Allocate Memory on GPU");
        goto FreeData;
    }

    status = cudaMemcpy(dev_mat, mat, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Copy Memory");
        goto FreeData;
    }

    status = cudaMemcpy(dev_pro, pro, size * sizeof(double), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Copy Memory");
        goto FreeData;
    }

    status = cudaMemcpy(dev_att, pro, size * sizeof(double), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Copy Memory");
        goto FreeData;
    }

    matFromProAtt << <size, size >> > (dev_mat, dev_res, dev_pro, dev_att, size);

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Launch Kernel");
        goto FreeData;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Error with a block");
        goto FreeData;
    }

    status = cudaMemcpy(res,dev_res,size * size * sizeof(int),cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Copy Memory From GPU");
        goto FreeData;
    }

FreeData:

    cudaFree(dev_att);
    cudaFree(dev_mat);
    cudaFree(dev_pro);
    cudaFree(dev_res);

    status = cudaDeviceReset();
    if (status != cudaSuccess)
    {
        fprintf(stderr, "Can't Reset Cuda");
    }

    return res;
}

