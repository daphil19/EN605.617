
__kernel void hello_kernel(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] + b[gid];
}

__kernel void add_kernel(__global const float *a, __global const float *b, __global float *result) {
    int gid = get_global_id(0);

    result[gid] = a[gid] + b[gid];
}

__kernel void sub_kernel(__global const float *a, __global const float *b, __global float *result) {
    int gid = get_global_id(0);

    result[gid] = a[gid] - b[gid];
}

__kernel void mult_kernel(__global const float *a, __global const float *b, __global float *result) {
    int gid = get_global_id(0);

    result[gid] = a[gid] * b[gid];
}

__kernel void div_kernel(__global const float *a, __global const float *b, __global float *result) {
    int gid = get_global_id(0);

    result[gid] = a[gid] / b[gid];
}

// TODO I need to do the pow
__kernel void pow_kernel(__global const float *a, __global const float *b, __global float *result) {
    int gid = get_global_id(0);

    result[gid] = pow(a[gid], b[gid]);
    // result[gid] = a[gid] + b[gid];
}

