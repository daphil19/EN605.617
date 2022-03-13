#ifndef operations_h
#define operations_h

__device__ unsigned int get_thread_index();
__global__ void add(unsigned int *res, unsigned int *first, unsigned int *second);
__global__ void subtract(unsigned int *res, unsigned int *first, unsigned int *second);
__global__ void mult(unsigned int *res, unsigned int *first, unsigned int *second);
__global__ void mod(unsigned int *res, unsigned int *first, unsigned int *second);

#endif