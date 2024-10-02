#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory.h>

// 打印当前主机下所有GPU设备的相关信息
void printDeviceInform() {
	cudaDeviceProp prop;

	// 获得设备数量
	int count;
	cudaGetDeviceCount(&count);

	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("--- General information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execition timeout: ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("--- Memory Information for device %d ---\n", i);
		printf("Total global mem: %zu\n", prop.totalGlobalMem);
		printf("Total constant mem: %zu\n", prop.totalConstMem);
		printf("Max mem pitch: %zu\n", prop.memPitch);
		printf("Texture Alignment: %zu\n", prop.textureAlignment);
		printf("--- MP Information for device %d ---\n", i);
		printf("Mutiprocessor count: %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp: %d\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max threads dimensions: (%d, %d, %d)\n",
			prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max gird dimentions:(%d, %d, %d)\n",
			prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
	}
}

// 根据指定要求选择设备
void chooseProperDevice() {
	cudaDeviceProp prop;
	int dev;
	cudaGetDevice(&dev);
	printf("ID of current CUDA device: %d\n", dev);
	memset(&prop, 0, sizeof(cudaDeviceProp));;
	prop.major = 1;
	prop.minor = 3;
	cudaChooseDevice(&dev, &prop);
	printf("ID of CUDA device closest to reversion 1.3:%d\n", dev);
	cudaSetDevice(dev);
}


int main() {

	/*printDeviceInform();*/

	chooseProperDevice();
}