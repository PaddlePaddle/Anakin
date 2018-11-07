#include "core/tensor.h"
#include "core/env.h"
#ifdef USE_CUBLAS
const char* cublas_get_errorstring(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "Unknown cublas status";
}

#endif


#ifdef USE_CUDNN
const char* cudnn_get_errorstring(cudnnStatus_t status) {
    switch (status) {
    case CUDNN_STATUS_SUCCESS:
        return "CUDNN_STATUS_SUCCESS";

    case CUDNN_STATUS_NOT_INITIALIZED:
        return "CUDNN_STATUS_NOT_INITIALIZED";

    case CUDNN_STATUS_ALLOC_FAILED:
        return "CUDNN_STATUS_ALLOC_FAILED";

    case CUDNN_STATUS_BAD_PARAM:
        return "CUDNN_STATUS_BAD_PARAM";

    case CUDNN_STATUS_INTERNAL_ERROR:
        return "CUDNN_STATUS_INTERNAL_ERROR";

    case CUDNN_STATUS_INVALID_VALUE:
        return "CUDNN_STATUS_INVALID_VALUE";

    case CUDNN_STATUS_ARCH_MISMATCH:
        return "CUDNN_STATUS_ARCH_MISMATCH";

    case CUDNN_STATUS_MAPPING_ERROR:
        return "CUDNN_STATUS_MAPPING_ERROR";

    case CUDNN_STATUS_EXECUTION_FAILED:
        return "CUDNN_STATUS_EXECUTION_FAILED";

    case CUDNN_STATUS_NOT_SUPPORTED:
        return "CUDNN_STATUS_NOT_SUPPORTED";

    case CUDNN_STATUS_LICENSE_ERROR:
        return "CUDNN_STATUS_LICENSE_ERROR";
#if CUDNN_VERSION_MIN(6, 0, 0)

    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
        return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
#endif
#if CUDNN_VERSION_MIN(7, 0, 0)

    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
        return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";

    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
        return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
#endif
    }

    return "Unknown cudnn status";
}
#endif

namespace anakin {

namespace saber {

#ifdef USE_CUDA

typedef TargetWrapper<NVHX86, __host_target> NVH_API;

void NVH_API::get_device_count(int& count) {
    //todo
    LOG(WARNING) << "host target NVHX86 \" get_device_count\" is not implemented";
    count = 1;
}

void NVH_API::set_device(int id) {
    //todo
    LOG(WARNING) << "host target NVHX86 \" set_device\" is not implemented";
}

void NVH_API::mem_alloc(void** ptr, size_t n) {
    CUDA_CHECK(cudaMallocHost(ptr, n));
}

void NVH_API::mem_free(void* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

void NVH_API::mem_set(void* ptr, int value, size_t n) {
    memset(ptr, value, n);
}

void NVH_API::create_event(event_t* event, bool flag) {
    if (flag) {
        CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDefault));
    } else {
        CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    }
}

void NVH_API::destroy_event(event_t event) {
    CUDA_CHECK(cudaEventDestroy(event));
}

void NVH_API::record_event(event_t event, stream_t stream) {
    CUDA_CHECK(cudaEventRecord(event, stream));
}

void NVH_API::create_stream(stream_t* stream) {
    CUDA_CHECK(cudaStreamCreate(stream));
}

void NVH_API::create_stream_with_flag(stream_t* stream, unsigned int flag) {
    CUDA_CHECK(cudaStreamCreateWithFlags(stream, flag));
}

void NVH_API::create_stream_with_priority(stream_t* stream, unsigned int flag, int priority) {
    CUDA_CHECK(cudaStreamCreateWithPriority(stream, flag, priority));
}

void NVH_API::destroy_stream(stream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void NVH_API::query_event(event_t event) {
    CUDA_CHECK(cudaEventQuery(event));
}

void NVH_API::sync_event(event_t event) {
    CUDA_CHECK(cudaEventSynchronize(event));
}

void NVH_API::sync_stream(event_t event, stream_t stream) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
}

void NVH_API::sync_stream(stream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void NVH_API::sync_memcpy(void* dst, size_t dst_offset, int dst_id, \
                          const void* src, size_t src_offset, int src_id, \
                          size_t count, __HtoH) {
    CUDA_CHECK(cudaMemcpy((char*)dst + dst_offset, (char*)src + src_offset, count,
                          cudaMemcpyHostToHost));
    CUDA_CHECK(cudaStreamSynchronize(0));
    //LOG(INFO) << "NVH, sync, H2H, size: " << count << ", src_offset: " \
            << src_offset << ", data:" << ((const float*)((char*)src + src_offset))[0];
}

void NVH_API::async_memcpy(void* dst, size_t dst_offset, int dst_id, \
                           const void* src, size_t src_offset, int src_id, \
                           size_t count, stream_t stream, __HtoH) {
    CUDA_CHECK(cudaMemcpy((char*)dst + dst_offset, (char*)src + src_offset, count,
                          cudaMemcpyHostToHost));
    //LOG(INFO) << "NVH, sync, H2H, size: " << count;
}

void NVH_API::sync_memcpy_p2p(void* dst, size_t dst_offset, int dst_id, \
                              const void* src, size_t src_offset, int src_id, size_t count) {}

void NVH_API::async_memcpy_p2p(void* dst, size_t dst_offset, int dst_id, \
                               const void* src, size_t src_offset, int src_id, \
                               size_t count, stream_t stream) {}

int NVH_API::get_device_id() {
    return 0;
}

void NVH_API::device_sync() {
    CUDA_CHECK(cudaDeviceSynchronize());
}
/**
 * \brief for NV device target only, device target is NV gpu
 * use cuda api to manage memory
 * support device to device, device to host, host to device memcpy
*/
typedef TargetWrapper<NV, __device_target> NV_API;

void NV_API::get_device_count(int& count) {
    CUDA_CHECK(cudaGetDeviceCount(&count));
}

void NV_API::set_device(int id) {
    CUDA_CHECK(cudaSetDevice(id));
}

void NV_API::mem_alloc(void** ptr, size_t n) {
    CUDA_CHECK(cudaMalloc(ptr, n));
}

void NV_API::mem_free(void* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void NV_API::mem_set(void* ptr, int value, size_t n) {
    CUDA_CHECK(cudaMemset(ptr, value, n));
}

void NV_API::create_event(event_t* event, bool flag) {
    if (flag) {
        CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDefault));
    } else {
        CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    }
}

void NV_API::create_stream(stream_t* stream) {
    CUDA_CHECK(cudaStreamCreate(stream));
}

/**
 * \brief create cuda stream with flag
 * @param stream    input stream
 * @param flag      input flag, 0: default stream flag, 1: cudaStreamNonBlocking
 */
void NV_API::create_stream_with_flag(stream_t* stream, unsigned int flag) {
    CUDA_CHECK(cudaStreamCreateWithFlags(stream, flag));
}

void NV_API::create_stream_with_priority(stream_t* stream, unsigned int flag, int priority) {
    CUDA_CHECK(cudaStreamCreateWithPriority(stream, flag, priority));
}

void NV_API::destroy_stream(stream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void NV_API::destroy_event(event_t event) {
    CUDA_CHECK(cudaEventDestroy(event));
}

void NV_API::record_event(event_t event, stream_t stream) {
    CUDA_CHECK(cudaEventRecord(event, stream));
}

void NV_API::query_event(event_t event) {
    CUDA_CHECK(cudaEventQuery(event));
}

void NV_API::sync_event(event_t event) {
    CUDA_CHECK(cudaEventSynchronize(event));
}

void NV_API::sync_stream(event_t event, stream_t stream) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
}

void NV_API::sync_stream(stream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void NV_API::sync_memcpy(void* dst, size_t dst_offset, int dst_id, \
                         const void* src, size_t src_offset, int src_id, \
                         size_t count, __DtoD) {

    if (dst_id == src_id) {
        CUDA_CHECK(cudaMemcpy((char*)dst + dst_offset, (char*)src + src_offset, count,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaStreamSynchronize(0));
        //LOG(INFO) << "cuda, sync, D2D, size: " << count;
    } else {
        CUDA_CHECK(cudaMemcpyPeer((char*)dst + dst_offset, dst_id, (char*)src + src_offset, src_id, count));
        //LOG(INFO) << "cuda, async, P2P, size: " << count;
    }
}

void NV_API::async_memcpy(void* dst, size_t dst_offset, int dst_id, \
                          const void* src, size_t src_offset, int src_id, \
                          size_t count, stream_t stream, __DtoD) {

    if (dst_id == src_id) {
        CUDA_CHECK(cudaMemcpyAsync((char*)dst + dst_offset, (char*)src + src_offset, count,
                                   cudaMemcpyDeviceToDevice, stream));
        //record_event(event, stream);
        //LOG(INFO) << "cuda, async, D2D, size: " << count;
    } else {
        CUDA_CHECK(cudaMemcpyPeerAsync((char*)dst + dst_offset, dst_id, (char*)src + src_offset, src_id,
                                       count, stream));
        //record_event(event, stream);
        //LOG(INFO) << "cuda, async P2P, size: " << count;
    }
}


void NV_API::sync_memcpy(void* dst, size_t dst_offset, int dst_id, \
                         const void* src, size_t src_offset, int src_id, \
                         size_t count, __HtoD) {
    CUDA_CHECK(cudaMemcpy((char*)dst + dst_offset, (char*)src + src_offset, count,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(0));
    //LOG(INFO) << "cuda, sync, H2D, size: " << count;
}

void NV_API::async_memcpy(void* dst, size_t dst_offset, int dst_id, \
                          const void* src, size_t src_offset, int src_id, \
                          size_t count, stream_t stream, __HtoD) {
    CUDA_CHECK(cudaMemcpyAsync((char*)dst + dst_offset, (char*)src + src_offset, count,
                               cudaMemcpyHostToDevice, stream));
    //record_event(event, stream);
    //LOG(INFO) << "cuda, async, H2D, size: " << count;
}

void NV_API::sync_memcpy(void* dst, size_t dst_offset, int dst_id, \
                         const void* src, size_t src_offset, int src_id, \
                         size_t count, __DtoH) {
    CUDA_CHECK(cudaMemcpy((char*)dst + dst_offset, (char*)src + src_offset, count,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(0));
    //LOG(INFO) << "cuda, sync, D2H, size: " << count;
}

void NV_API::async_memcpy(void* dst, size_t dst_offset, int dst_id, \
                          const void* src, size_t src_offset, int src_id, \
                          size_t count, stream_t stream, __DtoH) {
    CUDA_CHECK(cudaMemcpyAsync((char*)dst + dst_offset, (char*)src + src_offset, count,
                               cudaMemcpyDeviceToHost, stream));
    //record_event(event, stream);
    //LOG(INFO) << "cuda, async, D2H, size: " << count;
}

void NV_API::sync_memcpy_p2p(void* dst, size_t dst_offset, int dst_id, \
                             const void* src, size_t src_offset, int src_id, \
                             size_t count) {
    CUDA_CHECK(cudaMemcpyPeer((char*)dst + dst_offset, dst_id, (char*)src + src_offset, src_id, count));
    //LOG(INFO) << "cuda, sync, P2P, size: " << count;
}

void NV_API::async_memcpy_p2p(void* dst, size_t dst_offset, int dst_id, \
                              const void* src, size_t src_offset, int src_id, \
                              size_t count, stream_t stream) {
    CUDA_CHECK(cudaMemcpyPeerAsync((char*)dst + dst_offset, dst_id, (char*)src + src_offset, src_id,
                                   count, stream));
    //record_event(event, stream);
    //LOG(INFO) << "cuda, async, P2P, size: " << count;
}

/**
 * \brief device target return currently used device id
 * @return          currently activated device id
 */
int NV_API::get_device_id() {
    int device_id;
    cudaGetDevice(&device_id);
    return device_id;
}

void NV_API::device_sync() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

//! NV Buffer
template class Buffer<NV>;
template class Buffer<NVHX86>;

template class Tensor<NV>;
template class Tensor<NVHX86>;


//!
template struct Env<NV>;
template struct Env<NVHX86>;

#endif //USE_CUDA

} //namespace saber

} //namespace anakin
