
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

# include <stdio.h>

#define N 10

__global__ void add(int* a, int* b, int* c) {
	int tid = blockIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main() {
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)& dev_a, N * sizeof(int));
	cudaMalloc((void**)& dev_b, N * sizeof(int));
	cudaMalloc((void**)& dev_c, N * sizeof(int));
	
	//同样的为数组赋值的操作也可以搬到GPU上
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	
	// 尖括号内调用的格式是<<<girdDim, blockDim>>>
	// gridDim代表启动的网格（gird）中块（block）的数量
	// blockDim代表每个块中线程（thread）的数量
	// 如下的调用则证明一共是N*1=N个线程并行执行
	add << <N, 1 >> > (dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}