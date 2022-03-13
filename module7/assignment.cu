#include<stdio.h>
#include "operations.cuh"

typedef struct {
    unsigned int *firstInput;
    unsigned int *secondInput;
    unsigned int *output;
} OPERATION_ARRAYS_T;

__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

__host__ void print_delta(cudaEvent_t start, cudaEvent_t stop) {
    // TODO
    cudaEventSynchronize(stop);

	float delta = 0;
	cudaEventElapsedTime(&delta, start, stop);
    printf("%f\n", delta);
}

int main(int argc, char **argv)
{
	// read command line arguments
	unsigned int totalThreads = (1 << 20);
	unsigned int blockSize = 256;

	if (argc >= 2)
	{
		totalThreads = atoi(argv[1]);
	}
	else
	{
		printf("Using default total threads %d\n", totalThreads);
	}
	if (argc >= 3)
	{
		blockSize = atoi(argv[2]);
	}
	else
	{
		printf("Using default block size %d\n", blockSize);
	}

	unsigned int numBlocks = totalThreads / blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0)
	{
		++numBlocks;
		totalThreads = numBlocks * blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	size_t dataSizeBytes = sizeof(unsigned int) * totalThreads;

    // Initialize streams
    cudaStream_t streamAdd, streamSub, streamMul, streamMod;
    cudaStreamCreate(&streamAdd);
    cudaStreamCreate(&streamSub);
    cudaStreamCreate(&streamMul);
    cudaStreamCreate(&streamMod);

    // initialize timing events
    // cudaEvent_t startStream, stopStream, startPage, stopPage;


    // define and allocate arryas
    // TODO putting all of this into an array may make the most sense...
    unsigned int *firstInputCpu, *secondInputCpu;
    unsigned int *firstInputGpuAdd, *secondInputGpuAdd, *outputGpuAdd;
    unsigned int *firstInputGpuSub, *secondInputGpuSub, *outputGpuSub;
    unsigned int *firstInputGpuMul, *secondInputGpuMul, *outputGpuMul;
    unsigned int *firstInputGpuMod, *secondInputGpuMod, *outputGpuMod;
    unsigned int *firstInputGpu, *secondInputGpu, *outputGpu; // non-streamed processing

    cudaHostAlloc(&firstInputCpu, dataSizeBytes, cudaHostAllocDefault);
    cudaHostAlloc(&secondInputCpu, dataSizeBytes, cudaHostAllocDefault);

    cudaMalloc((void**)&firstInputGpuAdd, dataSizeBytes);
    cudaMalloc((void**)&secondInputGpuAdd, dataSizeBytes);
    cudaMalloc((void**)&outputGpuAdd, dataSizeBytes);

    cudaMalloc((void**)&firstInputGpuSub, dataSizeBytes);
    cudaMalloc((void**)&secondInputGpuSub, dataSizeBytes);
    cudaMalloc((void**)&outputGpuSub, dataSizeBytes);

    cudaMalloc((void**)&firstInputGpuMul, dataSizeBytes);
    cudaMalloc((void**)&secondInputGpuMul, dataSizeBytes);
    cudaMalloc((void**)&outputGpuMul, dataSizeBytes);

    cudaMalloc((void**)&firstInputGpuMod, dataSizeBytes);
    cudaMalloc((void**)&secondInputGpuMod, dataSizeBytes);
    cudaMalloc((void**)&outputGpuMod, dataSizeBytes);

    cudaMalloc((void**)&firstInputGpu, dataSizeBytes);
    cudaMalloc((void**)&secondInputGpu, dataSizeBytes);
    cudaMalloc((void**)&outputGpu, dataSizeBytes);


    // initialize array inputs
    for (int i = 0; i < totalThreads; i++) {
        firstInputCpu[i] = i;
        secondInputCpu[i] = rand();
    }

    // cudaEventCreate(&start)
    cudaEvent_t streamStart = get_time();

    // copy data onto gpu
    // TODO apparantly we async copy to the stream and then call the kernel
    cudaMemcpyAsync(firstInputGpuAdd, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamAdd);
    cudaMemcpyAsync(secondInputGpuAdd, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamAdd);

    cudaMemcpyAsync(firstInputGpuSub, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamSub);
    cudaMemcpyAsync(secondInputGpuSub, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamSub);

    cudaMemcpyAsync(firstInputGpuMul, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamMul);
    cudaMemcpyAsync(secondInputGpuMul, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamMul);

    cudaMemcpyAsync(firstInputGpuMod, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamMod);
    cudaMemcpyAsync(secondInputGpuMod, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamMod);


    // block before we process, just to make sure the copy is done in time
    cudaStreamSynchronize(streamAdd);
    add<<<numBlocks, blockSize, 0, streamAdd>>>(outputGpuAdd, firstInputGpuAdd, secondInputGpuAdd);
    cudaStreamSynchronize(streamAdd);

    cudaStreamSynchronize(streamSub);
    subtract<<<numBlocks, blockSize, 0, streamAdd>>>(outputGpuAdd, firstInputGpuAdd, secondInputGpuAdd);
    cudaStreamSynchronize(streamSub);

    cudaStreamSynchronize(streamMul);
    mult<<<numBlocks, blockSize, 0, streamAdd>>>(outputGpuAdd, firstInputGpuAdd, secondInputGpuAdd);
    cudaStreamSynchronize(streamMul);

    cudaStreamSynchronize(streamMod);
    mod<<<numBlocks, blockSize, 0, streamAdd>>>(outputGpuAdd, firstInputGpuAdd, secondInputGpuAdd);
    cudaStreamSynchronize(streamMod);

    cudaEvent_t streamStop = get_time();
    print_delta(streamStart, streamStop);

    // TODO output!

    cudaEvent_t syncStart = get_time();

    // yes, these copies are redundant, but their purpose is to emulate the operations above to provide a benchmark for performance
    cudaMemcpy(firstInputGpu, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(secondInputGpu, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice);
    add<<<numBlocks, blockSize>>>(outputGpu, firstInputGpu, secondInputGpu);
    cudaMemcpy(firstInputGpu, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(secondInputGpu, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice);
    subtract<<<numBlocks, blockSize>>>(outputGpu, firstInputGpu, secondInputGpu);
    cudaMemcpy(firstInputGpu, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(secondInputGpu, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice);
    mult<<<numBlocks, blockSize>>>(outputGpu, firstInputGpu, secondInputGpu);
    mod<<<numBlocks, blockSize>>>(outputGpu, firstInputGpu, secondInputGpu);
    cudaMemcpy(firstInputGpu, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(secondInputGpu, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice);

    cudaEvent_t syncStop = get_time();
    print_delta(syncStart, syncStop);


    cudaFreeHost(firstInputCpu);
    cudaFreeHost(secondInputCpu);

    cudaFree(firstInputGpuAdd);
    cudaFree(secondInputGpuAdd);
    cudaFree(outputGpuAdd);

    cudaFree(firstInputGpuSub);
    cudaFree(secondInputGpuSub);
    cudaFree(outputGpuSub);

    cudaFree(firstInputGpuMul);
    cudaFree(secondInputGpuMul);
    cudaFree(outputGpuMul);

    cudaFree(firstInputGpuMod);
    cudaFree(secondInputGpuMod);
    cudaFree(outputGpuMod);

    cudaFree(firstInputGpu);
    cudaFree(secondInputGpu);
    cudaFree(outputGpu);

    return EXIT_SUCCESS;
}
