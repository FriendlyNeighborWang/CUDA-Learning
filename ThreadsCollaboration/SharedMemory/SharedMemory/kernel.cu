#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#include <stdio.h>
#include <iostream>
#include "../common/book.h"

#define imin(a,b) (a<b?a:b)

const int N = 33 * 2014;
const int threadsPerBlock = 256;

// 计算点积
__global__ void dot(float* a, float* b, float* c) {
	// 声明一个在所有线程中共享的内存，多个线程都可以对该区域更改
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < N) {
		temp += a[tid] * b[tid];
		tid += blockDim.x + gridDim.x;
	}
	cache[cacheIndex] = temp;
	// 下面这个语句保证调用的线程同步，在这个语句之前多线程一定结束
	// 注：这个语句本身的要求是启用的所有线程都要调用一次才可以
	// 也就是说如果出现只需要一部分线程执行特定任务的线程发散情况时，尽量不要把语句放进线程的条件判断分支中
	__syncthreads();

	// 归约运算（求和运算）
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < 1) 
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	// 写入内存
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

constexpr int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

int main() {
	float* a, * b, c, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;

	a = new float[N];
	b = new float[N];
	partial_c = new float[blocksPerGrid];

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

	dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++) {
		c += partial_c[i];
	}
	std::cout << c << std::endl;

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
}