// Based on the work of Andrew Krepps

// NOTE formatting is based on the default formatter configuration of VS code

#include <stdio.h>
#include <stdlib.h>

#include "operations.cuh"

#define RANDOM_RANGE 4

// TODO if I had thought about this more I probably would have used a class instead of structs

typedef struct {
	unsigned int * firstInputCpu;
	unsigned int * secondInputCpu;
	unsigned int * firstInputGpu;
	unsigned int * secondInputGpu;
} INPUT_ARRAYS_T;

typedef struct {
	unsigned int *operationResultGpu;
	unsigned int *addResultCpu;
	unsigned int *subtractResultCpu;
	unsigned int *multResultCpu;
	unsigned int *modResultCpu;
} OUTPUT_ARRAYS_T;

typedef struct {
	unsigned int totalThreads;
	unsigned int blockSize;
	unsigned int numBlocks;
	size_t dataSizeBytes;
} INPUT_PARAMS_T;

void allocate_paged_inputs(INPUT_PARAMS_T* inputParams, INPUT_ARRAYS_T* input, OUTPUT_ARRAYS_T * output) {
		input->firstInputCpu = new unsigned int[inputParams->totalThreads];
		input->secondInputCpu = new unsigned int[inputParams->totalThreads]; 
		cudaMalloc((void **)&(input->firstInputGpu), inputParams->dataSizeBytes);
		cudaMalloc((void **)&(input->secondInputGpu), inputParams->dataSizeBytes);

		cudaMalloc((void**)&(output->operationResultGpu), inputParams->dataSizeBytes);
		output->addResultCpu = new unsigned int[inputParams->totalThreads];
		output->subtractResultCpu = new unsigned int[inputParams->totalThreads];
		output->multResultCpu = new unsigned int[inputParams->totalThreads];
		output->modResultCpu = new unsigned int[inputParams->totalThreads];

}

void cleanup_paged_inputs(INPUT_ARRAYS_T* input, OUTPUT_ARRAYS_T* output) {
	delete[] input->firstInputCpu;
	delete[] input->secondInputCpu;
	cudaFree(input->firstInputGpu);
	cudaFree(input->secondInputGpu);

	cudaFree(output->operationResultGpu);
	delete[] output->addResultCpu;
	delete[] output->subtractResultCpu;
	delete[] output->multResultCpu;
	delete[] output->modResultCpu;
}

void allocate_pinned_inputs(INPUT_PARAMS_T* inputParams, INPUT_ARRAYS_T* input, OUTPUT_ARRAYS_T* output) {
		cudaMallocHost((void **)&(input->firstInputCpu), inputParams->dataSizeBytes);
		cudaMallocHost((void **)&(input->secondInputCpu), inputParams->dataSizeBytes);
		cudaMalloc((void **)&(input->firstInputGpu), inputParams->dataSizeBytes);
		cudaMalloc((void **)&(input->secondInputGpu), inputParams->dataSizeBytes);

		cudaMalloc((void**)&(output->operationResultGpu), inputParams->dataSizeBytes);
		cudaMallocHost((void **)&(output->addResultCpu), inputParams->dataSizeBytes);
		cudaMallocHost((void **)&(output->subtractResultCpu), inputParams->dataSizeBytes);
		cudaMallocHost((void **)&(output->multResultCpu), inputParams->dataSizeBytes);
		cudaMallocHost((void **)&(output->modResultCpu), inputParams->dataSizeBytes);
}

void cleanup_pinned_inputs(INPUT_ARRAYS_T* input, OUTPUT_ARRAYS_T* output) {
	cudaFreeHost(input->firstInputCpu);
	cudaFreeHost(input->secondInputCpu);
	cudaFree(input->firstInputGpu);
	cudaFree(input->secondInputGpu);

	cudaFree(output->operationResultGpu);
	cudaFreeHost(output->addResultCpu);
	cudaFreeHost(output->subtractResultCpu);
	cudaFreeHost(output->multResultCpu);
	cudaFreeHost(output->modResultCpu);
}

void initialize_inputs(INPUT_PARAMS_T* inputParams, INPUT_ARRAYS_T* input) {
		for (int i = 0; i < inputParams->totalThreads; i++) {
		input->firstInputCpu[i] = i;
		input->secondInputCpu[i] = rand() % RANDOM_RANGE;
	}
}

__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

float perform_operations(INPUT_PARAMS_T * inputParams, INPUT_ARRAYS_T* input, OUTPUT_ARRAYS_T* output) {
	cudaEvent_t start_time = get_time();


	cudaMemcpy(input->firstInputGpu, input->firstInputCpu, inputParams->dataSizeBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(input->secondInputGpu, input->secondInputCpu, inputParams->dataSizeBytes, cudaMemcpyHostToDevice);


	add<<<inputParams->numBlocks, inputParams->blockSize>>>(output->operationResultGpu, input->firstInputGpu, input->secondInputGpu);

	cudaMemcpy(output->addResultCpu, output->operationResultGpu, inputParams->dataSizeBytes, cudaMemcpyDeviceToHost);

	for (unsigned int i = 0; i < inputParams->totalThreads; i++)
	{
		printf("%d + %d = %d\n", input->firstInputCpu[i], input->secondInputCpu[i], output->addResultCpu[i]);
	}

	subtract<<<inputParams->numBlocks, inputParams->blockSize>>>(output->operationResultGpu, input->firstInputGpu, input->secondInputGpu);
	cudaMemcpy(output->subtractResultCpu, output->operationResultGpu, inputParams->dataSizeBytes, cudaMemcpyDeviceToHost);

	for (unsigned int i = 0; i < inputParams->totalThreads; i++)
	{
		printf("%d - %d = %d\n", input->firstInputCpu[i], input->secondInputCpu[i], output->subtractResultCpu[i]);
	}

	mult<<<inputParams->numBlocks, inputParams->blockSize>>>(output->operationResultGpu, input->firstInputGpu, input->secondInputGpu);
	cudaMemcpy(output->multResultCpu, output->operationResultGpu, inputParams->dataSizeBytes, cudaMemcpyDeviceToHost);

	for (unsigned int i = 0; i < inputParams->totalThreads; i++)
	{
		printf("%d * %d = %d\n", input->firstInputCpu[i], input->secondInputCpu[i], output->multResultCpu[i]);
	}

	mod<<<inputParams->numBlocks, inputParams->blockSize>>>(output->operationResultGpu, input->firstInputGpu, input->secondInputGpu);
	cudaMemcpy(output->modResultCpu, output->operationResultGpu, inputParams->dataSizeBytes, cudaMemcpyDeviceToHost);

	for (unsigned int i = 0; i < inputParams->totalThreads; i++)
	{
		printf("%d %% %d = %d\n", input->firstInputCpu[i], input->secondInputCpu[i], output->modResultCpu[i]);
	}

		cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);
	return delta;

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

	INPUT_PARAMS_T inputParams = {
	totalThreads,
	blockSize,
	numBlocks,
	dataSizeBytes,
	};

	INPUT_ARRAYS_T input;

	OUTPUT_ARRAYS_T output;

	allocate_paged_inputs(&inputParams, &input, &output);
	initialize_inputs(&inputParams, &input);
	printf("Executing paged operations...");
	float pagedDelta = perform_operations(&inputParams, &input, &output);
	cleanup_paged_inputs(&input, &output);

	allocate_pinned_inputs(&inputParams, &input, &output);
	initialize_inputs(&inputParams, &input);
	printf("Executing pinned oeprations...");
	float pinnedDelta = perform_operations(&inputParams, &input, &output);
	cleanup_pinned_inputs(&input, &output);

	printf("Paged operations too %f ms\n", pagedDelta);
	printf("Pinned operations took %f ms\n", pinnedDelta);

	return EXIT_SUCCESS;
}
