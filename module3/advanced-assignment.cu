#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>

__device__ unsigned int get_thread_index()
{
    return (blockIdx.x * blockDim.x) + threadIdx.x;
}

__global__ void determine_greatest_even_multiple(unsigned int *block, unsigned int *results)
{
    const unsigned int thread_idx = get_thread_index();
    if (block[thread_idx] % 2 == 0)
    {
        results[thread_idx] = 1;
    }
}

bool sorter(unsigned int a, unsigned int b)
{
    if (a % 2 == 0 && b % 2 == 0)
    {
        return a <= b ? a : b;
    }
    else if (a % 2 == 0)
    {
        return a;
    }
    else if (b % 2 == 0)
    {
        return b;
    }

    return a <= b ? a : b;
}

int main(int argc, char **argv)
{
    unsigned int arraySize = (1 << 20); // 1MB
    unsigned int blockSize = 256;

    if (argc >= 2)
    {
        arraySize = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        blockSize = atoi(argv[2]);
    }

    int numBlocks = arraySize / blockSize;

    // validate command line arguments
    if (arraySize % blockSize != 0)
    {
        ++numBlocks;
        arraySize = numBlocks * blockSize;

        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", arraySize);
    }

    size_t dataSizeBytes = sizeof(unsigned int) * arraySize;

    curandGenerator_t rng;
    curandCreateGeneratorHost(&rng, CURAND_RNG_PSEUDO_DEFAULT);

    unsigned int *input;
    unsigned int *output;

    unsigned int *inputCPU = (unsigned int *)malloc(dataSizeBytes);

    cudaMalloc((void **)&input, dataSizeBytes);
    cudaMalloc((void **)&output, dataSizeBytes);

    // while this code is invoked from the host, it actually is run on device
    curandGenerate(rng, inputCPU, arraySize);
    cudaMemcpy(input, inputCPU, dataSizeBytes, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    // I needed to get this many iterations in order to actually get a measurable difference in time
    for (int i = 0; i < 100000; i++)
    {
        determine_greatest_even_multiple<<<numBlocks, blockSize>>>(input, output);
    }
    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << " Time elapsed GPU = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ns\n";

    unsigned int *optimizedCPU = (unsigned int *)malloc(dataSizeBytes);
    unsigned int startIdx = 0;
    unsigned int endIdx = arraySize - 1;

    for (int i = 0; i < arraySize; i++)
    {
        if (inputCPU[i] % 2 == 0)
        {
            optimizedCPU[startIdx++] = inputCPU[i];
        }
        else
        {
            optimizedCPU[endIdx--] = inputCPU[i];
        }
    }

    cudaMemcpy(input, optimizedCPU, dataSizeBytes, cudaMemcpyHostToDevice);

    start = std::chrono::high_resolution_clock::now();
    // I needed to get this many iterations in order for things to go even with having
    for (int i = 0; i < 100000; i++)
    {
        determine_greatest_even_multiple<<<numBlocks, blockSize>>>(input, output);
    }
    stop = std::chrono::high_resolution_clock::now();

    std::cout << " Time elapsed GPU = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ns\n";

    curandDestroyGenerator(rng);

    cudaFree(input);
    cudaFree(output);
    free(inputCPU);
    free(optimizedCPU);

    return EXIT_SUCCESS;
}