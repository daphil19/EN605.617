#include <stdio.h>

// struct that contains the CPU and GPU input arrays
typedef struct
{
	unsigned int *firstInputCpu;
	unsigned int *secondInputCpu;
	unsigned int *firstInputGpu;
	unsigned int *secondInputGpu;
} INPUT_ARRAYS_T;

// struct that contains the CPU and GPU output arrays
typedef struct
{
	unsigned int *cpuOutputBuffer;
	unsigned int *addResult;
	unsigned int *subtractResult;
	unsigned int *multResult;
	unsigned int *modResult;
} OUTPUT_ARRAYS_T;

// struct that contains the input parameters
typedef struct
{
	unsigned int totalThreads;
	unsigned int blockSize;
	unsigned int numBlocks;
	size_t dataSizeBytes;
} INPUT_PARAMS_T;

static const int RANDOM_RANGE = 4;

// gets the current thread index
__device__ unsigned int get_thread_index()
{
	return (blockIdx.x * blockDim.x) + threadIdx.x;
}

// in order to effectively leverage registered memory, we'll perform all operations at once
__global__ void perform_operations_global(unsigned int *firstInput, unsigned int *secondInput, OUTPUT_ARRAYS_T output)
{
	unsigned int idx = get_thread_index();
	output.addResult[idx] = firstInput[idx] + secondInput[idx];
	output.subtractResult[idx] = firstInput[idx] - secondInput[idx];
	output.multResult[idx] = firstInput[idx] * secondInput[idx];
	output.modResult[idx] = firstInput[idx] % secondInput[idx];
}

// in order to effectively leverage registered memory, we'll perform all operations at once
__global__ void perform_operations_register(unsigned int *firstInput, unsigned int *secondInput, OUTPUT_ARRAYS_T output)
{
	unsigned int idx = get_thread_index();
	unsigned int firstNum = firstInput[idx];
	unsigned int secondNum = secondInput[idx];
	output.addResult[idx] = firstNum + secondNum;
	output.subtractResult[idx] = firstNum - secondNum;
	output.multResult[idx] = firstNum * secondNum;
	output.modResult[idx] = firstNum % secondNum;
}

__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

void initialize_inputs(INPUT_PARAMS_T *inputParams, INPUT_ARRAYS_T *input, OUTPUT_ARRAYS_T *output)
{
	input->firstInputCpu = new unsigned int[inputParams->totalThreads];
	input->secondInputCpu = new unsigned int[inputParams->totalThreads];
	cudaMalloc((void **)&(input->firstInputGpu), inputParams->dataSizeBytes);
	cudaMalloc((void **)&(input->secondInputGpu), inputParams->dataSizeBytes);

	for (int i = 0; i < inputParams->totalThreads; i++)
	{
		input->firstInputCpu[i] = i;
		input->secondInputCpu[i] = rand() % RANDOM_RANGE;
	}

	cudaMemcpy(input->firstInputGpu, input->firstInputCpu, inputParams->dataSizeBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(input->secondInputGpu, input->secondInputCpu, inputParams->dataSizeBytes, cudaMemcpyHostToDevice);

	output->cpuOutputBuffer = new unsigned int[inputParams->totalThreads];
	cudaMalloc((void **)&(output->addResult), inputParams->dataSizeBytes);
	cudaMalloc((void **)&(output->subtractResult), inputParams->dataSizeBytes);
	cudaMalloc((void **)&(output->multResult), inputParams->dataSizeBytes);
	cudaMalloc((void **)&(output->modResult), inputParams->dataSizeBytes);
}

void cleanup(INPUT_ARRAYS_T *inputs, OUTPUT_ARRAYS_T *outputs)
{
	delete[] inputs->firstInputCpu;
	delete[] inputs->secondInputCpu;
	cudaFree(inputs->firstInputGpu);
	cudaFree(inputs->secondInputGpu);

	delete[] outputs->cpuOutputBuffer;
	cudaFree(outputs->addResult);
	cudaFree(outputs->subtractResult);
	cudaFree(outputs->multResult);
	cudaFree(outputs->modResult);
}

void cleanup_const(INPUT_ARRAYS_T *inputs, OUTPUT_ARRAYS_T *outputs)
{
	delete[] inputs->firstInputCpu;
	delete[] inputs->secondInputCpu;

	delete[] outputs->cpuOutputBuffer;
	cudaFree(outputs->addResult);
	cudaFree(outputs->subtractResult);
	cudaFree(outputs->multResult);
	cudaFree(outputs->modResult);
}

void print_results(INPUT_PARAMS_T inputParams, INPUT_ARRAYS_T input, OUTPUT_ARRAYS_T output)
{
	cudaMemcpy(output.cpuOutputBuffer, output.addResult, inputParams.dataSizeBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < inputParams.totalThreads; i++)
	{
		printf("%d + %d = %d\n", input.firstInputCpu[i], input.secondInputCpu[i], output.cpuOutputBuffer[i]);
	}

	cudaMemcpy(output.cpuOutputBuffer, output.subtractResult, inputParams.dataSizeBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < inputParams.totalThreads; i++)
	{
		printf("%d - %d = %d\n", input.firstInputCpu[i], input.secondInputCpu[i], output.cpuOutputBuffer[i]);
	}

	cudaMemcpy(output.cpuOutputBuffer, output.multResult, inputParams.dataSizeBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < inputParams.totalThreads; i++)
	{
		printf("%d * %d = %d\n", input.firstInputCpu[i], input.secondInputCpu[i], output.cpuOutputBuffer[i]);
	}

	cudaMemcpy(output.cpuOutputBuffer, output.modResult, inputParams.dataSizeBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < inputParams.totalThreads; i++)
	{
		printf("%d %% %d = %d\n", input.firstInputCpu[i], input.secondInputCpu[i], output.cpuOutputBuffer[i]);
	}
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

	initialize_inputs(&inputParams, &input, &output);

	cudaEvent_t start_time = get_time();
	// TODO I thought the shared memory size should be based on block size, but that seemed to break?
	perform_operations_global<<<numBlocks, blockSize>>>(input.firstInputGpu, input.secondInputGpu, output);
	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float globalDelta = 0;
	cudaEventElapsedTime(&globalDelta, start_time, end_time);

	print_results(inputParams, input, output);

	cleanup(&input, &output);

	initialize_inputs(&inputParams, &input, &output);

	start_time = get_time();
	perform_operations_register<<<numBlocks, blockSize>>>(input.firstInputGpu, input.secondInputGpu, output);
	end_time = get_time();
	cudaEventSynchronize(end_time);

	float registerDelta = 0;
	cudaEventElapsedTime(&registerDelta, start_time, end_time);

	print_results(inputParams, input, output);

	cleanup(&input, &output);

	printf("Global memory execution time: %f ms\n", globalDelta);
	printf("Register memory execution time: %f ms\n", registerDelta);

	return EXIT_SUCCESS;
}
