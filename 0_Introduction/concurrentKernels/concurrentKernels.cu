#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "helper_cuda.h"

namespace cg = cooperative_groups;


__global__ void clock_block(clock_t *d_output, clock_t clock_count) {
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count) {
        clock_t end_clock = clock();
        clock_offset = end_clock - start_clock;
    }
    d_output[0] = clock_offset;
}

template <typename T>
__inline__ __device__ T warpSumReduce(T value) {
    value += __shfl_down_sync(0xFFFFFFFF, value , 16);
    value += __shfl_down_sync(0xFFFFFFFF, value , 8);
    value += __shfl_down_sync(0xFFFFFFFF, value , 4);
    value += __shfl_down_sync(0xFFFFFFFF, value , 2);
    value += __shfl_down_sync(0xFFFFFFFF, value , 1);
    return value;
}

__global__ void sum(clock_t *d_clocks, int N) {
    cg::thread_block cta = cg::this_thread_block();

    __shared__ int shared[32];

    int value = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        value += d_clocks[i];
    }
    value = warpSumReduce<int>(value);

    int laneId = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    if (laneId == 0) shared[warpId] = value;
    cta.sync();
    value = (threadIdx.x < blockDim.x / warpSize) ? shared[laneId] : 0;     // assert blockDim.x % warpSize == 0
    if(warpId == 0) value = warpSumReduce<int>(value);

    if(threadIdx.x == 0) d_clocks[blockIdx.x] = value;
}

int main(int argc, char* argv[]) {
    // hyp
    int nkernels = 64;
    int nstreams = nkernels + 1;
    int nbytes = nkernels * sizeof(clock_t);
    float kernel_time = 10;

    // get device & info
    int devID = findCudaDevice(argc, (const char**)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    if (deviceProp.concurrentKernels == 0) {
        printf("> GPU does not support concurrent kernel execution\n");
        printf("  CUDA kernel runs will be serialized\n");
    }
    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // create stream
    cudaStream_t* streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    for(int i = 0; i < nstreams; i++) checkCudaErrors(cudaStreamCreate(&streams[i]));

    // create event
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));
    cudaEvent_t* events = (cudaEvent_t*) malloc(nkernels * sizeof(cudaEvent_t));
    for(int i = 0; i < nkernels; i++) checkCudaErrors(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));

    // malloc d_output
    clock_t *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_output, nbytes));
    
    // time
    clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
    clock_t total_clocks = time_clocks * nkernels;
    
    // work
    cudaEventRecord(start_event, 0);
    for (int i = 0; i < nkernels; ++i) {
        clock_block<<<1, 1, 0, streams[i]>>>(&d_output[i], time_clocks);
        checkCudaErrors(cudaEventRecord(events[i], streams[i]));
        checkCudaErrors(cudaStreamWaitEvent(streams[nstreams - 1], events[i], 0));
    }
    clock_t* output = (clock_t*)malloc(nbytes);
    checkCudaErrors(cudaMemcpyAsync(output, d_output, nbytes, cudaMemcpyDeviceToHost, streams[nstreams - 1]));
    sum<<<1, min((nkernels + 31) / 32 * 32, 1024), 0, streams[nstreams - 1]>>>(d_output, nkernels);
    clock_t output_sum;
    checkCudaErrors(cudaMemcpyAsync(&output_sum, d_output, sizeof(clock_t), cudaMemcpyDeviceToHost, streams[nstreams - 1]));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));

    // time
    float elapsed_time;
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

    printf("Expected time for serial execution of %d kernels = %.3fs\n", nkernels, nkernels * kernel_time / 1000.0f);
    printf("Expected time for concurrent execution of %d kernels = %.3fs\n", nkernels, kernel_time / 1000.0f);
    printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);

    // check
    assert(output_sum > total_clocks);
    clock_t output_sum2 = 0;
    for(int i = 0; i < nkernels; i++) output_sum2 += output[i];
    assert(output_sum == output_sum2);

    // free
    for (int i = 0; i < nkernels; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }
    cudaStreamDestroy(streams[nkernels]);
    free(streams);
    free(events);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_output);

    return 0;
}