
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../common/image.h"
#include "../common/book.h"

#define DIM 1024

struct DataBlock {
	unsigned char* dev_bitmap;
	IMAGE* bitmap;
};

// 释放在GPU上分配的内存
void cleanup(DataBlock* d) {
	cudaFree(d->dev_bitmap);
}

__global__ void kernel(unsigned char* ptr, int ticks) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.0f*cos(d/10.0f-ticks/7.0f) / (d / 10.0f + 1.0f));

	ptr[offset * 4 + 0] = grey;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

int main() {
	DataBlock data;
	IMAGE bitmap(DIM, DIM);
	data.bitmap = &bitmap;
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));

	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	int ticks = 0;
	bitmap.show_image(30);
	while (1) {
		kernel << <blocks, threads >> > (data.dev_bitmap, ticks);

		HANDLE_ERROR(cudaMemcpy(data.bitmap->get_ptr(), data.dev_bitmap, data.bitmap->image_size(), cudaMemcpyDeviceToHost));

		ticks++;
		char key = bitmap.show_image(30);
		if (key == 27) {
			break;
		}
	}
	cleanup(&data);
}