#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../common/book.h"
#include "../common/image.h"
#include <math.h>

#define INF 2e10f
#define DIM 1024
#define SPHERES 20
#define rnd(x) (x*rand()/RAND_MAX)

struct Sphere {
	float r, g, b;
	float radius;
	float x, y, z;
	__device__ float hit(float ox, float oy, float* n) {
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius) {
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

__global__ void kernel(unsigned char* ptr){
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int offset = x + y * blockDim.x + gridDim.x;

	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);
}

Sphere* s;

int main() {
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	
	IMAGE bitmap(DIM, DIM);
	unsigned char* dev_bitmap;
	auto sizeofSpheres = sizeof(Sphere) * SPHERES;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**)&s, sizeofSpheres));

	// 创建球体
	Sphere* temp_s = (Sphere*)malloc(sizeofSpheres);
	for (int i = 0; i < SPHERES; i++) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}

	HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeofSpheres, cudaMemcpyHostToDevice));
	free(temp_s);

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel<<<grids, threads>>>

}