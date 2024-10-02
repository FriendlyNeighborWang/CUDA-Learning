#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

// __global__标识符意味着该函数是可以由主机调用的设备函数
__global__ void add(int a, int b, int* c) {
	*c = a + b;
}

// 下面这段主要是将数据搬到了GPU上
int main(void) {
	int c;
	int* dev_c;
	// 在设备上分配内存
	cudaMalloc((void**)&dev_c, sizeof(int));
	add << <1, 1 >> > (2, 7, dev_c);
	// 将主机内存上的数据搬到GPU显存上
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("2+7 = %d\n", c);
	cudaFree(dev_c);
}