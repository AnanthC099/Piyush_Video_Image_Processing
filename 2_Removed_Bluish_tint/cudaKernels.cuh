#pragma once

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#ifdef __cplusplus
extern "C" {
#endif

	// Host function we call from C++ to do GPU processing on the shared texture.
	// In this "no effects" version, it does nothing except map/unmap for demonstration.
	void ApplyCudaKernel(cudaGraphicsResource* cudaResource, int width, int height);

#ifdef __cplusplus
}
#endif
