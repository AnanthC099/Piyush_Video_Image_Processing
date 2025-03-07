#include "cudaKernels.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

//------------------------------------------------------------------------------
// A do-nothing kernel
//------------------------------------------------------------------------------
__global__ void noOpKernel()
{
    // No operation
}

//------------------------------------------------------------------------------
// Map resource, then unmap, do nothing
//------------------------------------------------------------------------------
void ApplyCudaKernel(cudaGraphicsResource* cudaResource, int width, int height)
{
    if (!cudaResource) return;

    // Map the D3D11 resource into CUDA
    cudaError_t cuErr = cudaGraphicsMapResources(1, &cudaResource, 0);
    if (cuErr != cudaSuccess) {
        printf("cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(cuErr));
        return;
    }

    // Optionally run noOpKernel for demonstration:
    noOpKernel << <1, 1 >> > ();
    cudaDeviceSynchronize();

    // Unmap so D3D can use it again
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
}
