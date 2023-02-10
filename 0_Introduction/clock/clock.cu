#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#define NUM_BLOCKS 64
#define NUM_THREADS 512


template <typename T>
__inline__ __device__ T warpSumReduce(T value) {
    value += __shfl_down_sync(0xFFFFFFFF, value , 16);
    value += __shfl_down_sync(0xFFFFFFFF, value , 8);
    value += __shfl_down_sync(0xFFFFFFFF, value , 4);
    value += __shfl_down_sync(0xFFFFFFFF, value , 2);
    value += __shfl_down_sync(0xFFFFFFFF, value , 1);
    return value;
}

__global__ static void timedReduction( int* input, int* output, clock_t* timer) {
    extern __shared__ int shared[];
    if(threadIdx.x == 0) timer[blockIdx.x] = clock();

    int value = input[threadIdx.x];
    value = warpSumReduce<int>(value);

    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    if (laneId == 0) shared[warpId] = value;
    __syncthreads();
    value = (threadIdx.x < blockDim.x / warpSize) ? shared[laneId] : 0;     // assert blockDim.x % warpSize == 0
    if(warpId == 0) value = warpSumReduce<int>(value);

    if(threadIdx.x == 0) {
        output[blockIdx.x] = value;
        timer[blockIdx.x + gridDim.x] = clock();
    }
}


int main(int argc, char* argv[]) {
    findCudaDevice(argc, (const char**)argv);

    int* input = (int*)malloc(sizeof(int) * NUM_THREADS);
    for(int i = 0; i < NUM_THREADS; i++) input[i] = i;

    int* d_input, *d_output;
    checkCudaErrors(cudaMalloc((void**)&d_input, sizeof(int) * NUM_THREADS));
    checkCudaErrors(cudaMemcpy(d_input, input, sizeof(int) * NUM_THREADS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&d_output, sizeof(int) * NUM_BLOCKS));
    clock_t* d_timer;
    checkCudaErrors(cudaMalloc((void **)&d_timer, 2 * sizeof(clock_t) * NUM_BLOCKS));

    timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(int) * 32>>>(d_input, d_output, d_timer);

    clock_t* timer = (clock_t*)malloc(2 * sizeof(clock_t) * NUM_BLOCKS);
    checkCudaErrors(cudaMemcpy(timer, d_timer, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost));
    int* output = (int*)malloc(sizeof(int) * NUM_BLOCKS);
    checkCudaErrors(cudaMemcpy(output, d_output, sizeof(int) * NUM_BLOCKS, cudaMemcpyDeviceToHost));

    // check
    for(int i = 0; i < NUM_BLOCKS; i++) assert(output[i] == (NUM_THREADS - 1) * NUM_THREADS / 2);

    // avg time
    long double avgElapsedClocks = 0;
    for(int i = 0; i < NUM_BLOCKS; i++) {
        avgElapsedClocks += timer[i + NUM_BLOCKS] - timer[i];
    }
    avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf ms\n", avgElapsedClocks);


    free(input);
    free(output);
    free(timer);
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_timer));

    return 0;
}