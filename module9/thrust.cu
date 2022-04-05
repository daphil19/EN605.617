#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>

#include <iostream>

template <
    typename ThrustVectorType,
    typename BinaryFunction>
void compute_and_print(ThrustVectorType *first, ThrustVectorType *second, BinaryFunction op, char opStr, bool quiet)
{
    ThrustVectorType result(first->size());

    thrust::transform(first->begin(), first->end(), second->begin(), result.begin(), op);

    if (!quiet)
    {
        for (int i = 0; i < first->size(); i++)
        {
            std::cout << (*first)[i] << " " << opStr << " " << (*second)[i] << " = " << result[i] << std::endl;
        }
    }
}

template <
    typename ThrustVectorType>
void perform_operations(ThrustVectorType *first, ThrustVectorType *second, bool quiet)
{
    if (!quiet)
    {
        std::cout << "Add: " << std::endl;
    }
    compute_and_print(first, second, thrust::plus<int>(), '+', quiet);
    if (!quiet)
    {
        std::cout << "Substract: " << std::endl;
    }
    compute_and_print(first, second, thrust::minus<int>(), '-', quiet);
    if (!quiet)
    {
        std::cout << "Multiply: " << std::endl;
    }
    compute_and_print(first, second, thrust::multiplies<int>(), '*', quiet);
    if (!quiet)
    {
        std::cout << "Modulo: " << std::endl;
    }
    compute_and_print(first, second, thrust::modulus<int>(), '%', quiet);
}

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

int main(int argc, char const *argv[])
{

    size_t vector_size = 10;
    if (argc >= 2)
    {
        vector_size = atoi(argv[1]);
    }
    else
    {
        vector_size = 10;
        std::cout << "No vector size argument provided. A default of " << vector_size << " will be used." << std::endl;
    }

    // "quiet" flag. If provided, only the timings will be printed to terminal
    bool quiet = argc >= 3 && strncmp(argv[2], "--quiet", 7) == 0;

    /* code */
    thrust::device_vector<int> firstInputGpu(vector_size);

    thrust::sequence(firstInputGpu.begin(), firstInputGpu.end());

    // TODO instead of generating random number this way, we could look into
    // ways to generate the random numbers via curand and pass that to the
    // thrust constructor
    thrust::device_vector<int> secondInputGpu(vector_size);

    for (int i = 0; i < vector_size; i++)
    {
        secondInputGpu[i] = (rand() % 10) + 1; // gog a floating point exception with 0 for mod, so don't allow that as an option
    }

    cudaEvent_t device_start = get_time();
    perform_operations(&firstInputGpu, &secondInputGpu, quiet);
    cudaEvent_t device_end = get_time();

    thrust::host_vector<int> firstInputCpu(vector_size);
    thrust::host_vector<int> secondInputCpu(vector_size);

    thrust::copy(firstInputGpu.begin(), firstInputGpu.end(), firstInputCpu.begin());
    thrust::copy(secondInputGpu.begin(), secondInputGpu.end(), secondInputCpu.begin());

    cudaEvent_t host_start = get_time();
    perform_operations(&firstInputCpu, &secondInputCpu, quiet);
    cudaEvent_t host_end = get_time();

    std::cout << "GPU: ";
    print_delta(device_start, device_end);
    std::cout << "CPU: ";
    print_delta(host_start, host_end);

    return EXIT_SUCCESS;
}
