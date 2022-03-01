#include <stdio.h>

typedef struct
{
	unsigned int *firstInputCpu;
	unsigned int *secondInputCpu;
	unsigned int *firstInputGpu;
	unsigned int *secondInputGpu;
} INPUT_ARRAYS_T;

typedef struct
{
	unsigned int* cpuOutputBuffer;
	unsigned int *addResult;
	unsigned int *subtractResult;
	unsigned int *multResult;
	unsigned int *modResult;
} OUTPUT_ARRAYS_T;

typedef struct
{
	unsigned int totalThreads;
	unsigned int blockSize;
	unsigned int numBlocks;
	size_t dataSizeBytes;
} INPUT_PARAMS_T;

static const int RANDOM_RANGE = 4;
static const int CONST_ARRAY_SIZE = 8192;
static const int CONST_SIZE_BYTES = CONST_ARRAY_SIZE * sizeof(unsigned int);

__constant__ static unsigned int const_first_input[CONST_ARRAY_SIZE];
__constant__ static unsigned int const_second_input[CONST_ARRAY_SIZE];


__device__ unsigned int get_thread_index()
{
	return (blockIdx.x * blockDim.x) + threadIdx.x;
}


// in order to effectively leverage shared memory, we'll perform all operations at once
__global__ void perform_operations_shared(unsigned int* firstInput, unsigned int* secondInput, OUTPUT_ARRAYS_T output, unsigned int blockSize) {
    unsigned int idx = get_thread_index();

    // dynaic shared memory
    // NOTE that we have to put both sets of inputs into a single array... yay for index math!
    extern __shared__ unsigned int sharedInput[];

    // load inputs into shared memory
    sharedInput[idx] = firstInput[idx];

    // TODO do we need a protection here?
    sharedInput[idx + blockSize] = secondInput[idx];

	__syncthreads();

    // perform operations using shared memory
	output.addResult[idx] = sharedInput[idx] + sharedInput[idx + blockSize];
    output.subtractResult[idx] = sharedInput[idx] - sharedInput[idx + blockSize];
	output.multResult[idx] = sharedInput[idx] * sharedInput[idx + blockSize];
	output.modResult[idx] = sharedInput[idx] % sharedInput[idx + blockSize];

      __syncthreads();

}

__global__ void perform_operations_constant(OUTPUT_ARRAYS_T output) {
    unsigned int idx = get_thread_index();
	output.addResult[idx] = const_first_input[idx] + const_second_input[idx];
    output.subtractResult[idx] = const_first_input[idx] - const_second_input[idx];
    output.multResult[idx] = const_first_input[idx] * const_second_input[idx];
    output.modResult[idx] = const_first_input[idx] % const_second_input[idx];

}


__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

void initialize_inputs(INPUT_PARAMS_T *inputParams, INPUT_ARRAYS_T *input, OUTPUT_ARRAYS_T *output) {
	input->firstInputCpu = new unsigned int[inputParams->totalThreads];
	input->secondInputCpu = new unsigned int[inputParams->totalThreads];
	cudaMalloc((void **)&(input->firstInputGpu), inputParams->dataSizeBytes);
	cudaMalloc((void **)&(input->secondInputGpu), inputParams->dataSizeBytes);

	for (int i = 0; i < inputParams->totalThreads; i++) {
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

void initialize_constant_inputs(INPUT_ARRAYS_T *input, OUTPUT_ARRAYS_T *output) {
	input->firstInputCpu = new unsigned int[CONST_ARRAY_SIZE];
	input->secondInputCpu = new unsigned int[CONST_ARRAY_SIZE];

	for (int i = 0; i < CONST_ARRAY_SIZE; i++) {
		input->firstInputCpu[i] = i;
		input->secondInputCpu[i] = rand() % RANDOM_RANGE;
	}

	cudaMemcpyToSymbol(const_first_input, input->firstInputCpu, CONST_SIZE_BYTES);
	cudaMemcpyToSymbol(const_second_input, input->secondInputCpu, CONST_SIZE_BYTES);

	output->cpuOutputBuffer = new unsigned int[CONST_ARRAY_SIZE];
	cudaMalloc((void **)&(output->addResult), CONST_SIZE_BYTES);
	cudaMalloc((void **)&(output->subtractResult), CONST_SIZE_BYTES);
	cudaMalloc((void **)&(output->multResult), CONST_SIZE_BYTES);
	cudaMalloc((void **)&(output->modResult), CONST_SIZE_BYTES);


}

void cleanup(INPUT_ARRAYS_T *inputs, OUTPUT_ARRAYS_T* outputs) {
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

void cleanup_const(INPUT_ARRAYS_T *inputs, OUTPUT_ARRAYS_T* outputs) {
	delete[] inputs->firstInputCpu;
	delete[] inputs->secondInputCpu;

	delete[] outputs->cpuOutputBuffer;
	cudaFree(outputs->addResult);
	cudaFree(outputs->subtractResult);
	cudaFree(outputs->multResult);
	cudaFree(outputs->modResult);

}


int main(int argc, char* *argv) {

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
	perform_operations_shared<<<numBlocks, blockSize, totalThreads * sizeof(unsigned int) * 2>>>(input.firstInputGpu, input.secondInputGpu, output, blockSize);
	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float sharedDelta = 0;
	cudaEventElapsedTime(&sharedDelta, start_time, end_time);

	cudaMemcpy(output.cpuOutputBuffer, output.addResult, dataSizeBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < totalThreads; i++) {
		printf("%d + %d = %d\n", input.firstInputCpu[i], input.secondInputCpu[i], output.cpuOutputBuffer[i]);
	}

	cudaMemcpy(output.cpuOutputBuffer, output.subtractResult, dataSizeBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < totalThreads; i++) {
		printf("%d - %d = %d\n", input.firstInputCpu[i], input.secondInputCpu[i], output.cpuOutputBuffer[i]);
	}

	cudaMemcpy(output.cpuOutputBuffer, output.multResult, dataSizeBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < totalThreads; i++) {
		printf("%d * %d = %d\n", input.firstInputCpu[i], input.secondInputCpu[i], output.cpuOutputBuffer[i]);
	}

	cudaMemcpy(output.cpuOutputBuffer, output.modResult, dataSizeBytes, cudaMemcpyDeviceToHost);
	for (int i = 0; i < totalThreads; i++) {
		printf("%d %% %d = %d\n", input.firstInputCpu[i], input.secondInputCpu[i], output.cpuOutputBuffer[i]);
	}

	cleanup(&input, &output);

	INPUT_ARRAYS_T constInput;
	OUTPUT_ARRAYS_T constOutput;

	initialize_constant_inputs(&constInput, &constOutput);

    start_time = get_time();
    perform_operations_constant<<<inputParams.numBlocks, CONST_ARRAY_SIZE / inputParams.numBlocks>>>(constOutput);
    end_time = get_time();
	cudaEventSynchronize(end_time);

	float constantDelta = 0;
	cudaEventElapsedTime(&constantDelta, start_time, end_time);


	cudaMemcpy(constOutput.cpuOutputBuffer, constOutput.addResult, CONST_SIZE_BYTES, cudaMemcpyDeviceToHost);
	for (int i = 0; i < CONST_ARRAY_SIZE; i++) {
		printf("%d + %d = %d\n", constInput.firstInputCpu[i], constInput.secondInputCpu[i], constOutput.cpuOutputBuffer[i]);
	}

	cudaMemcpy(constOutput.cpuOutputBuffer, constOutput.subtractResult, CONST_SIZE_BYTES, cudaMemcpyDeviceToHost);
	for (int i = 0; i < CONST_ARRAY_SIZE; i++) {
		printf("%d - %d = %d\n", constInput.firstInputCpu[i], constInput.secondInputCpu[i], constOutput.cpuOutputBuffer[i]);
	}

	cudaMemcpy(constOutput.cpuOutputBuffer, constOutput.multResult, CONST_SIZE_BYTES, cudaMemcpyDeviceToHost);
	for (int i = 0; i < CONST_ARRAY_SIZE; i++) {
		printf("%d * %d = %d\n", constInput.firstInputCpu[i], constInput.secondInputCpu[i], constOutput.cpuOutputBuffer[i]);
	}

	cudaMemcpy(constOutput.cpuOutputBuffer, constOutput.modResult, CONST_SIZE_BYTES, cudaMemcpyDeviceToHost);
	for (int i = 0; i < CONST_ARRAY_SIZE; i++) {
		printf("%d %% %d = %d\n", constInput.firstInputCpu[i], constInput.secondInputCpu[i], constOutput.cpuOutputBuffer[i]);
	}

	cleanup_const(&constInput, &constOutput);


    printf("Shared memory execution time: %f ms\n", sharedDelta);
    printf("Constant memory execution time: %f ms\n", constantDelta);

	return EXIT_SUCCESS;
}
