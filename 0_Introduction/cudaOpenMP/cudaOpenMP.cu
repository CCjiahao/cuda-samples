#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"


__global__ void kernelAddConstant(int *g_a, const int b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_a[idx] += b;
}

int correctResult(int *data, const int n, const int b) {
  for (int i = 0; i < n; i++)
    if (data[i] != i + b) return 0;

  return 1;
}

int main(int argc, char *argv[]) {
    // get gpu & cpu numbers
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    assert(num_gpus > 0);
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);

    // get gpu info
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }
    printf("---------------------------\n");

    // cpu memory
    unsigned int n = num_gpus * 8192;
    unsigned int nbytes = n * sizeof(int);
    int* a = (int*)malloc(nbytes);
    for (unsigned int i = 0; i < n; i++) a[i] = i;

    // set num threads
    omp_set_num_threads(2*num_gpus);

#pragma omp parallel
    {
        // get threads id
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        // set device
        checkCudaErrors(cudaSetDevice(cpu_thread_id % num_gpus));
        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, cpu_thread_id % num_gpus);

        // split data
        int *sub_a = a + cpu_thread_id * n / num_cpu_threads;  // pointer to this CPU thread's portion of data
        unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
        int *d_a;
        checkCudaErrors(cudaMalloc((void **)&d_a, nbytes_per_kernel));
        checkCudaErrors(cudaMemcpy(d_a, sub_a, nbytes_per_kernel, cudaMemcpyHostToDevice));

        // work
        dim3 gpu_threads(128);  // 128 threads per block
        dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));
        kernelAddConstant<<<gpu_blocks, gpu_threads>>>(d_a, 3);

        // free
        checkCudaErrors(cudaMemcpy(sub_a, d_a, nbytes_per_kernel, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_a));
    }
    printf("---------------------------\n");

    // check error
    if(cudaSuccess != cudaGetLastError()) printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    assert(correctResult(a, n, 3));

    // free cpu memory
    free(a);

    return 0;
}
