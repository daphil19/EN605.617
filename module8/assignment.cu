#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cublas.h>
#include <cufft.h>


#define index(r,c,l) (((r)*(l))+(c))

static const int NO_OFFSET = 0;
static const int MAX = 100; // arbitrary value

__device__ unsigned int get_thread_index()
{
	return (blockIdx.x * blockDim.x) + threadIdx.x;
}

__global__ void init_random_number_states(unsigned int seed, curandState_t *states) {
    const unsigned int thread_idx = get_thread_index();

    curand_init(seed, thread_idx, NO_OFFSET, &states[thread_idx]);
}

__global__ void generate_random_numbers(curandState_t* states, unsigned int* numbers) {
    const unsigned int thread_idx = get_thread_index();
    numbers[thread_idx] = curand(&states[thread_idx]) % MAX;
}

int main(int argc, char **argv) {

    // TODO arguments


    int N = 10, M = 10;

    curandState_t* states;
    cudaMalloc((void**) &states, N * M * sizeof(curandState_t));

    unsigned int* gpu_A;
    unsigned int* gpu_B;
    // FIXME can't maintain this!
    unsigned int* cpu_A = new unsigned int[N * M];
    unsigned int* cpu_B = new unsigned int[M * N];


    cudaMalloc((void**) &gpu_A, N * M * sizeof(unsigned int));
    cudaMalloc((void**) &gpu_B, M * N * sizeof(unsigned int));


    init_random_number_states<<<N, M>>>(0, states);
    generate_random_numbers<<<N, M>>>(states, gpu_A);
    generate_random_numbers<<<M, N>>>(states, gpu_B);

    cudaMemcpy(cpu_A, gpu_A, N * M * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_B, gpu_B, N * M * sizeof(unsigned int), cudaMemcpyDeviceToHost);



    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("[%u, %u] -> %u\n", i, j, cpu_A[i * M + j]);
        }
    }

    cublasInit();

    // this is what we want to run when we end up running cublas
    // cublasSgemm('n','n',HA,WB,WA,1,AA,HA,BB,HB,0,CC,HC);


    cublasShutdown();

    cudaFree(states);
    cudaFree(gpu_A);
    cudaFree(gpu_B);

    delete[] cpu_A;
    delete[] cpu_B;


    cufftDoubleReal *signal;
    cufftDoubleComplex *freq_domain;
    cufftHandle plan;

    int fft_size = 16;
    size_t real_data_size_bytes = fft_size * sizeof(cufftDoubleReal);
    size_t complex_data_size_bytes = fft_size * sizeof(cufftDoubleComplex);
    cudaMalloc((void**) &signal, real_data_size_bytes);
    cudaMalloc((void**)&freq_domain, complex_data_size_bytes);

    // TODO is batch supposed to be used if we need to perform an fft that's smaller than the data?
    cufftPlan1d(&plan, fft_size, CUFFT_D2Z, 1);

    // out-of-place because we are doing a real-optimized fft
    // could do an in-place if we were doing a complex->complex fft
    cufftExecD2Z(plan, signal, freq_domain);

    // Perform FFT
    // cufftExecD2Z()

    cufftDestroy(plan);
    cudaFree(signal);
    cudaFree(freq_domain);

    return EXIT_SUCCESS;
}