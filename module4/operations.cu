#include "operations.cuh"

__device__ unsigned int get_thread_index()
{
	return (blockIdx.x * blockDim.x) + threadIdx.x;
}

__global__ void add(unsigned int *res, unsigned int *first, unsigned int *second)
{
	const unsigned int thread_idx = get_thread_index();
	res[thread_idx] = first[thread_idx] + second[thread_idx];
}

__global__ void subtract(unsigned int *res, unsigned int *first, unsigned int *second)
{
	// it would be great if we could create a device function that took a lambda to perform the operation
	const unsigned int thread_idx = get_thread_index();
	// NOTE it is possible that we will end up with some underflows here, especially in the first few indices
	res[thread_idx] = first[thread_idx] - second[thread_idx];
}

__global__ void mult(unsigned int *res, unsigned int *first, unsigned int *second)
{
	const unsigned int thread_idx = get_thread_index();
	res[thread_idx] = first[thread_idx] * second[thread_idx];
}

__global__ void mod(unsigned int *res, unsigned int *first, unsigned int *second)
{
	const unsigned int thread_idx = get_thread_index();
	res[thread_idx] = first[thread_idx] % second[thread_idx];
}
