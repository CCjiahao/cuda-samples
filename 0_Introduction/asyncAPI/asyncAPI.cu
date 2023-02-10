#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "helper_cuda.h"
#include "helper_timer.h"

__global__ void increment_kernel(int* g_data, int inc_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

bool correct_output(int *data, const int n, const int x) {
    for(int i = 0; i < n; i++) {
        if(data[i] != x) {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }
    }
    return true;
}


int main(int argc, char* argv[]) {
    // get device & get device info
    int devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    // hyp
    int n = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value = 26;

    // malloc host memory
    int *a = 0;
    checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
    memset(a, 0, nbytes);

    // malloc device memory
    int* d_a = 0;
    checkCudaErrors(cudaMalloc((void**)&d_a, nbytes));

    // create cuda event
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaDeviceSynchronize());

    // work and profiler
    checkCudaErrors(cudaProfilerStart());
    StopWatch timer;
    timer.start();
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    increment_kernel<<<n / 512, 512, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    timer.stop();
    checkCudaErrors(cudaProfilerStop());

    // record the time waiting for the device to complete its work
    unsigned long int counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    // device runtime
    float gpu_time;
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    printf("time spent executing by the GPU: %.2f ms\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f ms\n", timer.getTime());
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check answer
    assert(correct_output(a, n, value));

    // free
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));

    return 0;
}