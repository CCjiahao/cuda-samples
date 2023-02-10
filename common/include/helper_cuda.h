#ifndef __HELPER_CUDA_H__
#define __HELPER_CUDA_H__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "helper_string.h"

template <typename T>
void check(T result, char const *const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

inline void __getLastCudaError(const char* errorMessage, const char* file, const int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, static_cast<int>(err), cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline void __printLastCudaError(const char* errorMessage, const char* file, const int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, static_cast<int>(err), cudaGetErrorString(err));
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
            int SM;
            int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
        {0x50, 128}, {0x52, 128}, {0x53, 128},
        {0x60,  64}, {0x61, 128}, {0x62, 128},
        {0x70,  64}, {0x72,  64}, {0x75,  64},
        {0x80,  64}, {0x86, 128}, {0x87, 128},
        {0x90, 128},
        {-1, -1}
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

inline const char* _ConvertSMVer2ArchName(int major, int minor) {
    typedef struct {
        int SM;
        const char* name;
    } sSMtoArchName;

    sSMtoArchName nGpuArchNameSM[] = {
        {0x30, "Kepler"}, {0x32, "Kepler"}, {0x35, "Kepler"}, {0x37, "Kepler"},
        {0x50, "Maxwell"}, {0x52, "Maxwell"}, {0x53, "Maxwell"},
        {0x60, "Pascal"}, {0x61, "Pascal"}, {0x62, "Pascal"},
        {0x70, "Volta"}, {0x72, "Xavier"}, {0x75, "Turing"},
        {0x80, "Ampere"}, {0x86, "Ampere"}, {0x87, "Ampere"},
        {0x90, "Hopper"},
        {-1, "Graphics Device"}
    };

    int index = 0;
    while (nGpuArchNameSM[index].SM != -1) {
        if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchNameSM[index].name;
        }
        index++;
    }

    printf("MapSMtoArchName for SM %d.%d is undefined.  Default to use %s\n", major, minor, nGpuArchNameSM[index - 1].name);
    return nGpuArchNameSM[index - 1].name;
}

#ifdef __CUDA_RUNTIME_H__
inline int gpuDeviceInit(int devID) {
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (devID > device_count - 1 || devID < 0) {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", device_count);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        return -1;
    }

    int computeMode = -1, major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID));
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
    if (computeMode == cudaComputeModeProhibited) {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use cudaSetDevice().\n");
        return -1;
    }

    if (major < 1) {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaSetDevice(devID));
    printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, _ConvertSMVer2ArchName(major, minor));

    return devID;
}

inline int gpuGetMaxGflopsDeviceId() {
    int sm_per_multiproc = 0;
    int max_perf_device = 0;

    uint64_t max_compute_perf = 0;


    int device_count = 0, devices_prohibited = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    for (int current_device = 0; current_device < device_count; current_device++) {
        int computeMode = -1, major = 0, minor = 0;
        checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
        checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
        checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

        if (computeMode == cudaComputeModeProhibited) {
            devices_prohibited++;
            continue;
        }

        if (major == 9999 && minor == 9999) {
            sm_per_multiproc = 1;
        } else {
            sm_per_multiproc = _ConvertSMVer2Cores(major, minor);
        }
        int multiProcessorCount = 0, clockRate = 0;
        checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
        cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
        if (result != cudaSuccess) {
            if(result == cudaErrorInvalidValue) {
                clockRate = 1;
            }
            else {
                fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__, static_cast<unsigned int>(result), cudaGetErrorName(result));
                exit(EXIT_FAILURE);
            }
        }
        uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

        if (compute_perf > max_compute_perf) {
            max_compute_perf = compute_perf;
            max_perf_device = current_device;
        }

    }

    if (devices_prohibited == device_count) {
        fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.\n");
        exit(EXIT_FAILURE);
    }

    return max_perf_device;
}

inline int findCudaDevice(int argc, const char** argv) {
    int devID = 0;

    if(checkCmdLineFlag(argc, argv, "device")) {
        devID = getCmdLineArgumentInt(argc, argv, "device=");
        if (devID < 0) {
            printf("Invalid command line parameter\n ");
            exit(EXIT_FAILURE);
        } 
        if (gpuDeviceInit(devID) < 0) {
            printf("exiting...\n");
            exit(EXIT_FAILURE);
        }
    }
    else {
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors(cudaSetDevice(devID));
        int major = 0, minor = 0;
        checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
        checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, _ConvertSMVer2ArchName(major, minor), major, minor);
    }
    return devID;
}

#endif  // __CUDA_RUNTIME_H__

#endif  // __HELPER_CUDA_H__