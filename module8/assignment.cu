#include <iostream>

#include <curand.h>
#include <curand_kernel.h>
#include <cublas.h>
#include <cufft.h>

#define index(r,c,l) (((r)*(l))+(c))

static const int NO_OFFSET = 0;
static const int MAX = 10; // arbitrary value

__device__ unsigned int get_thread_index()
{
	return (blockIdx.x * blockDim.x) + threadIdx.x;
}

__global__ void init_random_number_states(unsigned int seed, curandState_t *states) {
    const unsigned int thread_idx = get_thread_index();

    curand_init(seed, thread_idx, NO_OFFSET, &states[thread_idx]);
}

__global__ void generate_random_numbers(curandState_t* states, double* numbers) {
    const unsigned int thread_idx = get_thread_index();
    numbers[thread_idx] = curand(&states[thread_idx]) % MAX;
}

__global__ void generate_cosine_wave(cufftDoubleReal* signal) {
    const unsigned int thread_idx = get_thread_index();
    // yes, this caues waprs, but we don't care as much about the performance
    // of this kernel as its just generating example data
    switch(thread_idx % 4) {
        case 0: signal[thread_idx] = 1; break;
        case 1: signal[thread_idx] = 0; break;
        case 2: signal[thread_idx] = -1; break;
        case 3: signal[thread_idx] = 0; break;
    }
}

int main(int argc, char **argv) {

    int N = 10, M = 10;

    std::cout << "Matrix Multiplication: (" << M << "x" << N << ") * (" << N << "x" << M << ")" << std::endl;

    double* cpu_A = new double[M*N];
    double* cpu_B = new double[N*M];
    double* cpu_C = new double[M*M];

    for (int i = 0; i < M * N; i++) {
        cpu_A[i] = rand() % MAX;
        cpu_B[i] = rand() % MAX;
    }

    for (int i = 0; i < M * M; i++) {
        cpu_C[i] = rand() % MAX;
    }

    double* gpu_A;
    double* gpu_B;
    double* gpu_C;
    cublasInit();

    cublasAlloc(M*N, sizeof(double), (void**)&gpu_A);
    cublasAlloc(N*M, sizeof(double), (void**)&gpu_B);
    cublasAlloc(M*M, sizeof(double), (void**)&gpu_C);

    cublasSetMatrix(M, N, sizeof(double), cpu_A, M, gpu_A, M);
    cublasSetMatrix(N, M, sizeof(double), cpu_B, N, gpu_B, N);


    cublasDgemm('n', 'n', M, M, N, 1, gpu_A, M, gpu_B, N, 0, gpu_C, M);

    cublasGetMatrix(M, M, sizeof(double), gpu_C, M, cpu_C, M);

    cublasFree(gpu_A);
    cublasFree(gpu_B);
    cublasFree(gpu_C);
    
    cublasShutdown();


    // print output
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << cpu_A[index(i, j, M)] << " ";
        }
        std::cout << std::endl;
    }

    // extra \n for better separation
    std::cout << std::endl;

    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << cpu_B[index(i, j, N)] << " ";
        }
        std::cout << std::endl;
    }

    // extra \n for better separation
    std::cout << std::endl;

    std::cout << "Matrix C (A x B);" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << cpu_C[index(i, j, M)] << " ";
        }
        std::cout << std::endl;
    }

    delete[] cpu_A;
    delete[] cpu_B;
    delete[] cpu_C;


    cufftDoubleReal *signal;
    cufftDoubleComplex *freq_domain;
    cufftHandle plan;

    int fft_size = 4;

    // extra \ns for better separation
    std::cout << std::endl << std::endl;

    std::cout << "Perform FFT of size " << fft_size << std::endl;

    size_t real_data_size_bytes = fft_size * sizeof(cufftDoubleReal);

    // for real->complex fft, the result size is dataSize (read: fftSize) / 2 + 1
    int results_size = (fft_size / 2) + 1;

    size_t complex_data_size_bytes = results_size * sizeof(cufftDoubleComplex);
    cudaMalloc((void**) &signal, real_data_size_bytes);
    cudaMalloc((void**)&freq_domain, complex_data_size_bytes);

    // TODO is this a good layout? or should we try to optimize better?
    generate_cosine_wave<<<fft_size, 1>>>(signal);

    // TODO is batch supposed to be used if we need to perform an fft that's smaller than the data?
    cufftPlan1d(&plan, fft_size, CUFFT_D2Z, 1);

    // out-of-place because we are doing a real-optimized fft
    // could do an in-place if we were doing a complex->complex fft
    cufftExecD2Z(plan, signal, freq_domain);

    // int results_size = (fft_size / 2) + 1;

    double2 *result = new double2[results_size];

    std::cout << "Input Singal" << std::endl;
    double *signal_cpu = new double[fft_size];
    cudaMemcpy(signal_cpu, signal, real_data_size_bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < fft_size; i++) {
        std::cout << signal_cpu[i] << std::endl;
    }  
    delete[] signal_cpu;

    // extra \n for better separation
    std::cout << std::endl;

    std::cout << "FFT output (dropping redundant data)" << std::endl;
    cudaMemcpy(result, freq_domain, complex_data_size_bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < results_size; i++) {
        std::cout << result[i].x << " + " << result[i].y << "j" << std::endl;
    }

    cufftDestroy(plan);
    cudaFree(signal);
    cudaFree(freq_domain);
    delete[] result;

    return EXIT_SUCCESS;
}
