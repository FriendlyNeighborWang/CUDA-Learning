#pragma once
#include "cuda_runtime.h"

struct cuComplex {
	float r;
	float i;
	__device__ cuComplex(float a, float b):r(a),i(b){}
	__device__ float magnituude2() {
		return r * r + i * i;
	}

	__device__ cuComplex operator*(const cuComplex& rhs) {
		return cuComplex(r * rhs.r - i * rhs.i, i * rhs.r + r * rhs.i);
	}
	__device__ cuComplex operator+(const cuComplex& rhs) {
		return cuComplex(r + rhs.r, i + rhs.i);
	}
};