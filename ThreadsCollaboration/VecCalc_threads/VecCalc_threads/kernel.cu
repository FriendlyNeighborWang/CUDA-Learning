
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../common/book.h"

#define N (33 * 1024)

// 递增步长所用到的blockDim和gridDim不是每个GPU上每个block和grid真正有多少元素
// 而是block上当前有多少被调用的threads和grid上有多少被调用的blocks
// 而这两个数值取决于调用add函数时尖括号中包含的数量
__global__ void add(int* a, int* b, int* c) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
		
}

int main() {
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * i;
	}
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
	
	// 此处代表每个grid上调用了128个blocks, 同时每个block上调用了128个threads
	add << < 128, 128 >> > (dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
	
	bool success = true;
	for (int i = 0; i < N; i++) {
		if (a[i] + b[i] != c[i]) {
			printf("%d + %d = %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}
	if (success)
		printf("We did it!");

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}