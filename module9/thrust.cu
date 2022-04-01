#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <iostream>



template<
typename ThrustVectorType,
typename BinaryFunction
>
void compute_and_print(ThrustVectorType *first, ThrustVectorType *second, BinaryFunction op, char opStr) {
    ThrustVectorType result(first->size());

    thrust::transform(first->begin(), first->end(), second->begin(), result.begin(), op);

    for (int i = 0; i < first->size(); i++) {
        std::cout << (*first)[i] << " " << opStr << " " << (*second)[i] << " = " << result[i] << std::endl;
    }
}

template<
typename ThrustVectorType
>
void perform_operations(ThrustVectorType *first, ThrustVectorType *second) {
    std::cout << "Add: " << std::endl;
    compute_and_print(first, second, thrust::plus<int>(), '+');
    std::cout << "Substract: " << std::endl;
    compute_and_print(first, second, thrust::minus<int>(), '-');
    std::cout << "Multiply: " << std::endl;
    compute_and_print(first, second, thrust::multiplies<int>(), '*');
    std::cout << "Modulo: " << std::endl;
    compute_and_print(first, second, thrust::modulus<int>(), '%');
}

int main(int argc, char const *argv[])
{

    size_t vector_size = 10;

    /* code */
    thrust::device_vector<int> firstInputGpu(vector_size);

    thrust::sequence(firstInputGpu.begin(), firstInputGpu.end());

    // thrust::device_vector<int> secondInput()

    // TODO instead of generating random number this way, we could look into
    // ways to generate the random numbers via curand and pass that to the
    // thrust constructor
    thrust::device_vector<int> secondInputGpu(vector_size);

    for (int i = 0; i < vector_size; i++) {
        secondInputGpu[i] = rand() % 10;
    }

    perform_operations(&firstInputGpu, &secondInputGpu);

    thrust::host_vector<int> firstInputCpu(vector_size);
    thrust::host_vector<int> secondInputCpu(vector_size);

    thrust::copy(firstInputGpu.begin(), firstInputGpu.end(), firstInputCpu.begin());
    thrust::copy(secondInputGpu.begin(), secondInputGpu.end(), secondInputCpu.begin());

    perform_operations(&firstInputCpu, &secondInputCpu);


    return EXIT_SUCCESS;
}
