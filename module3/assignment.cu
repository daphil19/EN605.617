// Based on the work of Andrew Krepps

// NOTE formatting is based on the default formatter configuration of VS code

#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#define RANDOM_RANGE 4

__device__ unsigned int get_thread_index()
{
	return (blockIdx.x * blockDim.x) + threadIdx.x;
}

// this kernel is responsible for initializing the values of the first operational array
// specifically, the value at a given index is the index itself
__global__ void populate_first_array(unsigned int *block)
{

	// FIXME we need to adjust this to properly handle multiple blocks...
	// I need to remember how to do that
	const unsigned int thread_idx = get_thread_index();
	block[thread_idx] = thread_idx;
}

// this kernel "normalizes" the random generated data
// "normalize" here means ensuring the bounds are within the defined range
__global__ void normalize_second_array(unsigned int *block)
{
	const unsigned int thread_idx = get_thread_index();
	block[thread_idx] = block[thread_idx] % RANDOM_RANGE;
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

int main(int argc, char **argv)
{

	// read command line arguments
	if (argc != 3)
	{
		printf("Must provide 2 command line arguments\n");
		return EXIT_FAILURE;
	}

	int totalThreads = atoi(argv[1]);
	int blockSize = atoi(argv[2]);
	int numBlocks = totalThreads / blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0)
	{
		++numBlocks;
		totalThreads = numBlocks * blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);

		// TODO we should probably fail here too
	}

	size_t dataSizeBytes = sizeof(unsigned int) * totalThreads;

	// TODO we need to create 3 arrays... one that contains numeric values in the array, and 1 with random values 0-3, and 1 for the results!
	unsigned int firstInputCpu[totalThreads];
	unsigned int secondInputCpu[totalThreads];
	unsigned int addResultCpu[totalThreads];
	unsigned int subtractResultCpu[totalThreads];
	unsigned int multResultCpu[totalThreads];
	unsigned int modResultCpu[totalThreads];

	unsigned int *firstInputGpu;
	unsigned int *secondInputGpu;
	unsigned int *operationResultGpu;

	// allocate cuda arrays
	cudaMalloc((void **)&firstInputGpu, dataSizeBytes);
	cudaMalloc((void **)&secondInputGpu, dataSizeBytes);
	cudaMalloc((void **)&operationResultGpu, dataSizeBytes);

	populate_first_array<<<numBlocks, blockSize>>>(firstInputGpu);

	curandGenerator_t rng;
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);

	// TODO the seed of rng is the same run-to-run
	// if we wanted it to be different, we should use the system time

	// while this code is invoked from the host, it actually is run on device
	curandGenerate(rng, secondInputGpu, totalThreads);
	// take the random numbers and transform them so that they are mod 3
	// while we could combine the generation and this into a single kernel, it seemed like a bit more configuration that I didn't want to deal with

	// clean up rng since we're done with it
	curandDestroyGenerator(rng);

	normalize_second_array<<<numBlocks, blockSize>>>(secondInputGpu);

	add<<<numBlocks, blockSize>>>(operationResultGpu, firstInputGpu, secondInputGpu);
	cudaMemcpy(addResultCpu, operationResultGpu, dataSizeBytes, cudaMemcpyDeviceToHost);

	subtract<<<numBlocks, blockSize>>>(operationResultGpu, firstInputGpu, secondInputGpu);
	cudaMemcpy(subtractResultCpu, operationResultGpu, dataSizeBytes, cudaMemcpyDeviceToHost);

	mult<<<numBlocks, blockSize>>>(operationResultGpu, firstInputGpu, secondInputGpu);
	cudaMemcpy(multResultCpu, operationResultGpu, dataSizeBytes, cudaMemcpyDeviceToHost);

	mod<<<numBlocks, blockSize>>>(operationResultGpu, firstInputGpu, secondInputGpu);
	cudaMemcpy(modResultCpu, operationResultGpu, dataSizeBytes, cudaMemcpyDeviceToHost);

	cudaFree(firstInputGpu);
	cudaFree(secondInputGpu);
	cudaFree(operationResultGpu);

	// for (unsigned int i = 0; i < totalThreads; i++) {
	// 	printf("%d -> %d\n", firstInputCpu[i], secondInputCpu[i]);

	// }

	return EXIT_SUCCESS;
}
