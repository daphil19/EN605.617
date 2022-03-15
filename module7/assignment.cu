#include <stdio.h>
#include "operations.cuh"

static const int RANDOM_RANGE = 4;

// struct for handling input and output arrays for a given operation
typedef struct
{
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

// print the delta based on the provided start and stop events
__host__ void print_delta(cudaEvent_t start, cudaEvent_t stop)
{
    cudaEventSynchronize(stop);

    float delta = 0;
    cudaEventElapsedTime(&delta, start, stop);
    printf("%f\n", delta);
}

// allocates **GPU** based arrays for an operation
__host__ OPERATION_ARRAYS_T initialize_operation_arrays(size_t dataSizeBytes)
{
    OPERATION_ARRAYS_T ops;
    cudaMalloc((void **)&ops.firstInput, dataSizeBytes);
    cudaMalloc((void **)&ops.secondInput, dataSizeBytes);
    cudaMalloc((void **)&ops.output, dataSizeBytes);

    return ops;
}

// frees the **GPU** based arrays for an operation
__host__ void free_operation_arrays(OPERATION_ARRAYS_T ops)
{
    cudaFree(ops.firstInput);
    cudaFree(ops.secondInput);
    cudaFree(ops.output);
}

// pretty-prints the results of a given set of operations inputs and output
__host__ void printResults(OPERATION_ARRAYS_T op, int size, char operation)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d %c %d = %d\n", op.firstInput[i], operation, op.secondInput[i], op.output[i]);
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
    // "quiet" flag. If provided, only the timings will be printed to terminal
    bool quiet = argc >= 4 && strncmp(argv[3], "--quiet", 7) == 0;

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

    // define and allocate arryas
    unsigned int *firstInputCpu, *secondInputCpu, *outputCpu;
    unsigned int *firstInputGpu, *secondInputGpu, *outputGpu; // non-streamed processing

    cudaHostAlloc(&firstInputCpu, dataSizeBytes, cudaHostAllocDefault);
    cudaHostAlloc(&secondInputCpu, dataSizeBytes, cudaHostAllocDefault);
    cudaHostAlloc(&outputCpu, dataSizeBytes, cudaHostAllocDefault);

    OPERATION_ARRAYS_T addOps = initialize_operation_arrays(dataSizeBytes);
    OPERATION_ARRAYS_T subOps = initialize_operation_arrays(dataSizeBytes);
    OPERATION_ARRAYS_T mulOps = initialize_operation_arrays(dataSizeBytes);
    OPERATION_ARRAYS_T modOps = initialize_operation_arrays(dataSizeBytes);

    cudaMalloc((void **)&firstInputGpu, dataSizeBytes);
    cudaMalloc((void **)&secondInputGpu, dataSizeBytes);
    cudaMalloc((void **)&outputGpu, dataSizeBytes);

    // initialize array inputs
    for (int i = 0; i < totalThreads; i++)
    {
        firstInputCpu[i] = i;
        secondInputCpu[i] = rand() % RANDOM_RANGE;
    }

    // cudaEventCreate(&start)
    cudaEvent_t streamStart = get_time();

    // copy data onto gpu
    cudaMemcpyAsync(addOps.firstInput, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamAdd);
    cudaMemcpyAsync(addOps.secondInput, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamAdd);

    cudaMemcpyAsync(subOps.firstInput, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamSub);
    cudaMemcpyAsync(subOps.secondInput, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamSub);

    cudaMemcpyAsync(mulOps.firstInput, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamMul);
    cudaMemcpyAsync(mulOps.secondInput, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamMul);

    cudaMemcpyAsync(modOps.firstInput, firstInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamMod);
    cudaMemcpyAsync(modOps.secondInput, secondInputCpu, dataSizeBytes, cudaMemcpyHostToDevice, streamMod);

    // block before we process, just to make sure the copy is done in time
    cudaStreamSynchronize(streamAdd);
    add<<<numBlocks, blockSize, 0, streamAdd>>>(addOps.output, addOps.firstInput, addOps.secondInput);
    cudaStreamSynchronize(streamAdd);

    cudaStreamSynchronize(streamSub);
    subtract<<<numBlocks, blockSize, 0, streamAdd>>>(subOps.output, subOps.firstInput, subOps.secondInput);
    cudaStreamSynchronize(streamSub);

    cudaStreamSynchronize(streamMul);
    mult<<<numBlocks, blockSize, 0, streamAdd>>>(mulOps.output, mulOps.firstInput, mulOps.secondInput);
    cudaStreamSynchronize(streamMul);

    cudaStreamSynchronize(streamMod);
    mod<<<numBlocks, blockSize, 0, streamAdd>>>(modOps.output, modOps.firstInput, modOps.secondInput);
    cudaStreamSynchronize(streamMod);

    cudaEvent_t streamStop = get_time();

    if (!quiet)
    {
        OPERATION_ARRAYS_T host_outputs = {
            firstInputCpu,
            secondInputCpu,
            outputCpu};
        cudaMemcpyAsync(host_outputs.output, addOps.output, dataSizeBytes, cudaMemcpyHostToDevice, streamAdd);
        // block to make sure the copy is done correctly
        cudaStreamSynchronize(streamAdd);
        printResults(host_outputs, totalThreads, '+');

        cudaMemcpyAsync(host_outputs.output, subOps.output, dataSizeBytes, cudaMemcpyHostToDevice, streamSub);
        cudaStreamSynchronize(streamSub);
        printResults(host_outputs, totalThreads, '-');

        cudaMemcpyAsync(host_outputs.output, mulOps.output, dataSizeBytes, cudaMemcpyHostToDevice, streamMul);
        cudaStreamSynchronize(streamMul);
        printResults(host_outputs, totalThreads, '*');

        cudaMemcpyAsync(host_outputs.output, modOps.output, dataSizeBytes, cudaMemcpyHostToDevice, streamMod);
        cudaStreamSynchronize(streamMod);
        printResults(host_outputs, totalThreads, '%');
    }

    printf("stream runtime: ");
    print_delta(streamStart, streamStop);

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

    // we won't output the synchronous operations because we already know those work correctly

    cudaEvent_t syncStop = get_time();
    printf("synchronous runtime: ");
    print_delta(syncStart, syncStop);

    cudaFreeHost(firstInputCpu);
    cudaFreeHost(secondInputCpu);
    cudaFreeHost(outputCpu);

    free_operation_arrays(addOps);
    free_operation_arrays(subOps);
    free_operation_arrays(mulOps);
    free_operation_arrays(modOps);

    cudaFree(firstInputGpu);
    cudaFree(secondInputGpu);
    cudaFree(outputGpu);

    return EXIT_SUCCESS;
}
