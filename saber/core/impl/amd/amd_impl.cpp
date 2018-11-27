#include "core/tensor.h"
#include "core/env.h"

namespace anakin{

namespace saber{

#ifdef AMD_GPU

#define AMD_GPU_EXTENSION

const char* opencl_get_error_string(cl_int err){
    switch (err) {
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    }
    return "Unknown cl error";
    
}


/**
 * \brief for AMD device target only, device target is AMD gpu
 * use opencl api to manage memory
 * support device to device, device to host, host to device memcpy
*/
typedef TargetWrapper<AMD, __device_target> AMD_API;
typedef Env<AMD> AMD_ENV;

int AMD_API::current_device_id_index = 0;
std::map<void *, cl_mem> AMD_API::buffers;

void AMD_API::get_device_count(int &count) {
    cl_platform_id id = AMD_ENV::get_platform_id();
    cl_uint nums;
    AMD_CHECK(clGetDeviceIDs(id, CL_DEVICE_TYPE_GPU, 0, NULL, &nums));
    count = (int)nums;
}

void AMD_API::set_device(int id){
    LOG(INFO) << "set device id = " << id;
    current_device_id_index = id;
}
        
void AMD_API::mem_alloc(TPtr* ptr, size_t n){
    AMD_ENV::is_init();

#ifdef  AMD_GPU_EXTENSION
    //LOG(INFO) << "use CL_MEM_USE_PERSISTENT_MEM_AMD to create buffer.";
#else
    //LOG(INFO) << "use CL_MEM_ALLOC_HOST_PTR to create buffer.";
#endif

    int index = get_device_id();

    cl_context context = AMD_ENV::cur_env()[index].get_context();

    cl_int err;
    cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE
#ifdef AMD_GPU_EXTENSION
        | CL_MEM_USE_PERSISTENT_MEM_AMD
#else
        | CL_MEM_ALLOC_HOST_PTR
#endif
           , n,  NULL, &err);

    AMD_CHECK(err);

    //ClMem* clbuf = (ClMem*)malloc(sizeof(ClMem));
    //clbuf->dmem = buf;
    //clbuf->offset = 0;
    ClMem clbuf(buf);

    //*ptr = (void *) buf;
    *ptr = clbuf;

    LOG(INFO) << __func__ << "device =" << index << " get context :" << context << " buffer :" << buf <<" size :" << n;
}

void AMD_API::mem_free(TPtr ptr){

    cl_mem mem = ptr.dmem;
    if (mem != nullptr) {
        clReleaseMemObject(mem);
    }
}

#if 1
void AMD_API::mem_set(TPtr ptr, int value, size_t n){

    cl_mem mem = ptr.dmem;

    if(mem == nullptr)
        return ;
    
    AMD_ENV::is_init();

    Device<AMD> dev = AMD_ENV::cur_env()[current_device_id_index];
    stream_t cm = dev.get_available_stream();

    clEnqueueFillBuffer(cm, mem, &value, sizeof(int), 0, n, 0, NULL, NULL);
}

#else

template<typename U>
void AMD_API::mem_set(TPtr ptr, U value, size_t n){
    if(ptr == nullptr)
        return ;
    
    AMD_ENV::is_init();

    Device<AMD> dev = AMD_ENV::cur_env()[current_device_id_index];
    stream_t cm = dev.get_available_stream(stream);

    cl_mem mem = ptr.dmem;

    clEnqueueFillBuffer(cm, mem, &value, sizeof(U), 0, n, 0, NULL, NULL);
}
#endif

void AMD_API::create_event(event_t& event, bool flag) {

    LOG(INFO) << "ceate_event break opencl call sequence. Is baidu expect clCreateUserEvent?";
    //do nothing for this.
    event = nullptr;

    //Env<AMD>::is_init();
    //cl_int err = CL_SUCCESS;
    //event = clCreaeUserEvent(AMD_ENV::cur_env()[current_device_id_index].context, &err);
    //AMD_CHECK(err);
}

void AMD_API::create_stream(stream_t& stream) {
    create_stream_with_flag(stream, 0);
}

/**
 * \brief create cuda stream with flag
 * @param stream    input stream
 * @param flag      input flag, 0: default stream flag, 1: cudaStreamNonBlocking
 */
void AMD_API::create_stream_with_flag(stream_t& stream, unsigned int flag) {
    Env<AMD>::is_init();
    cl_int err = CL_SUCCESS;
    stream  = clCreateCommandQueue(Env<AMD>::cur_env()[current_device_id_index].get_context(), Env<AMD>::cur_env()[current_device_id_index].get_device(), (cl_command_queue_properties) flag, &err);
    AMD_CHECK(err);
}

void AMD_API::_create_stream_with_flag(stream_t& stream, cl_context context, cl_device_id dev, unsigned int flag){
    cl_int err = CL_SUCCESS;
    stream  = clCreateCommandQueue(context, dev, (cl_command_queue_properties) flag, &err);
    AMD_CHECK(err);
}

void AMD_API::create_stream_with_priority(stream_t& stream, unsigned int flag, int priority) {
    // TODO
    LOG(ERROR) << "not support, use create_stream_with_flag to instead";
    create_stream_with_flag(stream, flag);
}

void AMD_API::destroy_stream(stream_t& stream) {
    AMD_CHECK(clReleaseCommandQueue(stream));
}

void AMD_API::destroy_event(event_t& event) {
//    LOG(INFO) << __func__ <<" :Does baidu expect this event is an User Event?";

    if(event == nullptr){
//        LOG(INFO) << "event is empty, do nothing";
        return;
    }

    cl_command_type t;
    AMD_CHECK(clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cl_command_type), &t, NULL));
    if ( t == CL_COMMAND_USER) {
        cl_int refs;
        AMD_CHECK(clGetEventInfo(event, CL_EVENT_REFERENCE_COUNT, sizeof(cl_int), &refs, NULL));
        if(refs == 1)
           AMD_CHECK(clSetUserEventStatus(event, CL_COMPLETE));

        AMD_CHECK(clReleaseEvent(event));
    } else {
//        LOG(INFO) << "NOT User Event, do nothing";
    }

}

void AMD_API::record_event(event_t& event, stream_t stream) {
    //LOG(WARNING) << "OpenCL record event when calling clEnqueueXXX, so we use marker to simulate this behavior";
    AMD_CHECK(clEnqueueMarkerWithWaitList(stream, 0, NULL, &event));
    //LOG(INFO) << "marker event "<< event;
}

void AMD_API::query_event(event_t& event) {
    // TODO
    LOG(ERROR) << "OpenCL us clGetEventInfo to retrive event's specific info. so we need to know what info user want to know";
}

void AMD_API::sync_event(event_t& event) {
//    LOG(INFO) << __func__ ;

    if(event == nullptr){
        LOG(INFO) << "event is empty, do nothing";
        return;
    }
    
//    LOG(INFO) << "sync_event E " << event;
    AMD_CHECK(clWaitForEvents( 1, &event));
//    LOG(INFO) << "sync_event X " << event;
}

void AMD_API::sync_stream(event_t& event, stream_t& stream) {
    LOG(INFO) << __func__ ;
    if(event != nullptr) {
        LOG(INFO) << "event is null";
        return;
    }

    LOG(INFO) << "sync_stream E ";
    AMD_CHECK(clEnqueueBarrierWithWaitList(stream, 1, &event, NULL));
    clFlush(stream);
    LOG(INFO) << "sync_stream D ";
}

        
void AMD_API::sync_memcpy(TPtr dst, int dst_id, const TPtr src, int src_id, \
        size_t count, __DtoD) {

    cl_mem dst_mem = dst.dmem;
    cl_mem src_mem = src.dmem;

    size_t dst_offset = dst.offset;
    size_t src_offset = src.offset;

    LOG(INFO)  << __func__<< " D2D dst=" << (void*)dst_mem << " dst_id=" << dst_id << " dst_office=" << dst_offset << " src=" <<(void*)src_mem << " src_id=" << src_id << " src_offset=" << src_offset << " count=" << count;
    //sync_memcpy_with_offset(dst, dst_id, 0, src, src_id, 0, count, __DtoD());

    if(dst_id == src_id){
        cl_command_queue cm = AMD_ENV::cur_env()[dst_id].get_available_stream();
        cl_event event;
        AMD_CHECK(clEnqueueCopyBuffer(cm, src_mem, dst_mem, src_offset, dst_offset, count, 0, NULL, &event));
        clFlush(cm);
        clWaitForEvents(1, &event);
        LOG(INFO) << "OpenCL, sync, D2D, size: " << count;
    } else{
        cl_command_queue dst_cm = AMD_ENV::cur_env()[dst_id].get_available_stream();
        cl_command_queue src_cm = AMD_ENV::cur_env()[src_id].get_available_stream();

        cl_int err;
        cl_event event;
        void *host_ptr = clEnqueueMapBuffer(src_cm, src_mem, CL_TRUE, CL_MAP_READ, src_offset, count, 0, NULL, NULL, &err);
        AMD_CHECK(err);
        AMD_CHECK(clEnqueueWriteBuffer(dst_cm, dst_mem, CL_TRUE, dst_offset, count, host_ptr, 0, NULL, NULL));
        AMD_CHECK(clEnqueueUnmapMemObject(src_cm, src_mem, host_ptr, 0, NULL, &event));
        clFlush(src_cm);
        clFlush(dst_cm);
        clWaitForEvents(1, &event);
        LOG(INFO) << "OpenCL, sync, P2P, size: " << count;
    }
}
#if 0
void AMD_API::sync_memcpy_with_offset(void* dst, int dst_id, size_t dst_offset, const void* src, int src_id, size_t src_offset, \
    // TODO
        size_t count, __DtoD) {

    LOG(INFO)  << __func__<< " D2D dst=" << dst << " dst_id=" << dst_id << " dst_office=" << dst_offset << " src=" <<src << " src_id=" << src_id << " src_offset=" << src_offset << " count=" << count;

    ClMem* clbuf_dst = (ClMem*)dst;
    ClMem* clbuf_src = (ClMem*)src;

    cl_mem dst_mem = dst.dmem;
    cl_mem src_mem = src.dmem;

    //cl_mem dst_mem = (cl_mem) dst;
    //cl_mem src_mem = (cl_mem) src;

    if(dst_id == src_id){
        cl_command_queue cm = AMD_ENV::cur_env()[dst_id].get_available_stream();
        cl_event event;
        AMD_CHECK(clEnqueueCopyBuffer(cm, src_mem, dst_mem, src_offset, dst_offset, count, 0, NULL, &event));
        clFlush(cm);
        clWaitForEvents(1, &event);
        LOG(INFO) << "OpenCL, sync, D2D, size: " << count;
    } else{
        cl_command_queue dst_cm = AMD_ENV::cur_env()[dst_id].get_available_stream();
        cl_command_queue src_cm = AMD_ENV::cur_env()[src_id].get_available_stream();

        cl_int err;
        cl_event event;
        void *host_ptr = clEnqueueMapBuffer(src_cm, src_mem, CL_TRUE, CL_MAP_READ, src_offset, count, 0, NULL, NULL, &err);
        AMD_CHECK(err);
        AMD_CHECK(clEnqueueWriteBuffer(dst_cm, dst_mem, CL_TRUE, dst_offset, count, host_ptr, 0, NULL, NULL));
        AMD_CHECK(clEnqueueUnmapMemObject(src_cm, src_mem, host_ptr, 0, NULL, &event));
        clFlush(src_cm);
        clFlush(dst_cm);
        clWaitForEvents(1, &event);
        LOG(INFO) << "OpenCL, sync, P2P, size: " << count;
    }

}
#endif
        
void AMD_API::async_memcpy(TPtr dst, int dst_id, const TPtr src, int src_id, \
        size_t count, stream_t &stream, __DtoD) {

    cl_mem dst_mem = dst.dmem;
    cl_mem src_mem = src.dmem;

    size_t dst_offset = dst.offset;
    size_t src_offset = src.offset;
    //async_memcpy_with_offset(dst, dst_id, 0, src, src_id, 0, count, stream, __DtoD());

    LOG(INFO)  << __func__<< " D2D dst=" << (void*)dst_mem << " dst_id=" << dst_id << " dst_office=" << dst_offset << " src=" << (void*)src_mem << " src_id=" << src_id << " src_offset=" << src_offset << " count=" << count;

    //cl_mem dst_mem = (cl_mem) dst;
    //cl_mem src_mem = (cl_mem) src;

    if(dst_id == src_id){
        cl_command_queue cm = AMD_ENV::cur_env()[dst_id].get_available_stream(stream);
        AMD_CHECK(clEnqueueCopyBuffer(cm, src_mem, dst_mem, src_offset, dst_offset, count, 0, NULL, NULL));
        clFlush(cm);
        LOG(INFO) << "OpenCL, sync, D2D, size: " << count;
    } else{
        cl_command_queue dst_cm = AMD_ENV::cur_env()[dst_id].get_available_stream(stream);
        cl_command_queue src_cm = AMD_ENV::cur_env()[src_id].get_available_stream(stream);

        cl_int err;
        cl_event dst_event, src_event, src_event2;
        void *host_ptr = clEnqueueMapBuffer(src_cm, src_mem, CL_FALSE, CL_MAP_READ, src_offset, count, 0, NULL, &dst_event, &err);
        AMD_CHECK(err);
        AMD_CHECK(clEnqueueWriteBuffer(dst_cm, dst_mem, CL_FALSE, dst_offset, count, host_ptr, 1, &dst_event, &src_event));
        AMD_CHECK(clEnqueueUnmapMemObject(src_cm, src_mem, host_ptr, 1, &src_event2, NULL));
        clFlush(src_cm);
        clFlush(dst_cm);
        LOG(INFO) << "OpenCL, sync, P2P, size: " << count;
    }

}
#if 0
void AMD_API::async_memcpy_with_offset(void* dst, int dst_id, size_t dst_offset, const void* src, int src_id, size_t src_offset, \
        size_t count, stream_t &stream, __DtoD) {

    LOG(INFO)  << __func__<< " D2D dst=" << dst << " dst_id=" << dst_id << " dst_office=" << dst_offset << " src=" <<src << " src_id=" << src_id << " src_offset=" << src_offset << " count=" << count;

    cl_mem dst_mem = (cl_mem) dst;
    cl_mem src_mem = (cl_mem) src;

    if(dst_id == src_id){
        cl_command_queue cm = AMD_ENV::cur_env()[dst_id].get_available_stream(stream);
        AMD_CHECK(clEnqueueCopyBuffer(cm, src_mem, dst_mem, src_offset, dst_offset, count, 0, NULL, NULL));
        clFlush(cm);
        LOG(INFO) << "OpenCL, sync, D2D, size: " << count;
    } else{
        cl_command_queue dst_cm = AMD_ENV::cur_env()[dst_id].get_available_stream(stream);
        cl_command_queue src_cm = AMD_ENV::cur_env()[src_id].get_available_stream(stream);

        cl_int err;
        cl_event dst_event, src_event, src_event2;
        void *host_ptr = clEnqueueMapBuffer(src_cm, src_mem, CL_FALSE, CL_MAP_READ, src_offset, count, 0, NULL, &dst_event, &err);
        AMD_CHECK(err);
        AMD_CHECK(clEnqueueWriteBuffer(dst_cm, dst_mem, CL_FALSE, dst_offset, count, host_ptr, 1, &dst_event, &src_event));
        AMD_CHECK(clEnqueueUnmapMemObject(src_cm, src_mem, host_ptr, 1, &src_event2, NULL));
        clFlush(src_cm);
        clFlush(dst_cm);
        LOG(INFO) << "OpenCL, sync, P2P, size: " << count;
    }

}
#endif

void AMD_API::sync_memcpy(TPtr dst, int dst_id, const void* src, int src_id, \
        size_t count, __HtoD) {

    cl_mem dst_mem = dst.dmem;
    size_t dst_offset = dst.offset;

    LOG(INFO)  << __func__<< " H2D dst=" << (void*)dst_mem << " dst_id=" << dst_id << " dst_office=" << dst_offset << " src=" <<src << " src_id=" << src_id << " count=" << count;

    cl_event event;
    cl_command_queue dst_cm = AMD_ENV::cur_env()[dst_id].get_available_stream();
    clEnqueueWriteBuffer(dst_cm, dst_mem, CL_TRUE, dst_offset, count, src, 0, NULL, &event);
    clFlush(dst_cm);
    clWaitForEvents(1, &event);
    LOG(INFO) << "OpenCL, sync, H2D, size: " << count;
}
#if 0
void AMD_API::sync_memcpy_with_offset(void* dst, int dst_id, size_t dst_offset, const void* src, int src_id, size_t src_offset, \
       size_t count, __HtoD) {

    LOG(INFO)  << __func__<< " H2D dst=" << dst << " dst_id=" << dst_id << " dst_office=" << dst_offset << " src=" <<src << " src_id=" << src_id << " src_offset=" << src_offset << " count=" << count;

    cl_event event;
    cl_mem dst_mem = (cl_mem) dst;
    cl_command_queue dst_cm = AMD_ENV::cur_env()[dst_id].get_available_stream();
    clEnqueueWriteBuffer(dst_cm, dst_mem, CL_TRUE, dst_offset, count, (char *)src + src_offset, 0, NULL, &event);
    clFlush(dst_cm);
    clWaitForEvents(1, &event);
    LOG(INFO) << "OpenCL, sync, H2D, size: " << count;

}
#endif
void AMD_API::async_memcpy(TPtr dst, int dst_id, const void* src, int src_id, \
        size_t count, stream_t& stream, __HtoD) {
    //async_memcpy_with_offset(dst, dst_id, 0, src, src_id, 0, count, stream, __HtoD());
    cl_mem dst_mem = dst.dmem;
    size_t dst_offset = dst.offset;

    LOG(INFO)  << __func__<< " H2D dst=" << (void*)dst_mem << " dst_id=" << dst_id << " dst_office=" << dst_offset << " src=" <<src << " src_id=" << src_id << " count=" << count;

    cl_command_queue dst_cm = AMD_ENV::cur_env()[dst_id].get_available_stream(stream);
    clEnqueueWriteBuffer(dst_cm, dst_mem, CL_FALSE, dst_offset, count, src, 0, NULL, NULL);
    clFlush(dst_cm);
    LOG(INFO) << "OpenCL, async, H2D, size: " << count;

}
#if 0
void AMD_API::async_memcpy_with_offset(void* dst, int dst_id, size_t dst_offset, const void* src, int src_id, size_t src_offset, \
        size_t count, stream_t& stream, __HtoD) {

    LOG(INFO)  << __func__<< " H2D dst=" << dst << " dst_id=" << dst_id << " dst_office=" << dst_offset << " src=" <<src << " src_id=" << src_id << " src_offset=" << src_offset << " count=" << count;

    cl_mem dst_mem = (cl_mem) dst;
    cl_command_queue dst_cm = AMD_ENV::cur_env()[dst_id].get_available_stream(stream);
    clEnqueueWriteBuffer(dst_cm, dst_mem, CL_FALSE, dst_offset, count, (char *)src + src_offset, 0, NULL, NULL);
    clFlush(dst_cm);
    LOG(INFO) << "OpenCL, async, H2D, size: " << count;
}
#endif
void AMD_API::sync_memcpy(void* dst, int dst_id, const TPtr src, int src_id, \
        size_t count, __DtoH) {
    //sync_memcpy_with_offset(dst, dst_id, 0, src, src_id, 0, count, __DtoH());
    cl_mem src_mem = src.dmem;
    size_t src_offset = src.offset;

    LOG(INFO)  << __func__<< " D2H dst=" << dst << " dst_id=" << dst_id << " src=" << (void*)src_mem << " src_id=" << src_id << " src_offset=" << src_offset << " count=" << count;

    cl_event event;
    cl_command_queue src_cm = AMD_ENV::cur_env()[src_id].get_available_stream();
    clEnqueueReadBuffer(src_cm, src_mem, CL_TRUE, src_offset, count, dst, 0, NULL, &event);
    clFlush(src_cm);
    clWaitForEvents(1, &event);
    LOG(INFO) << "OpenCL, sync, D2H, size: " << count;
}
#if 0
void AMD_API::sync_memcpy_with_offset(void* dst, int dst_id, size_t dst_offset, const void* src, int src_id, size_t src_offset, \
        size_t count, __DtoH) {
    LOG(INFO)  << __func__<< " D2H dst=" << dst << " dst_id=" << dst_id << " dst_office=" << dst_offset << " src=" <<src << " src_id=" << src_id << " src_offset=" << src_offset << " count=" << count;
    
    cl_event event;
    cl_mem src_mem = (cl_mem) src;
    cl_command_queue src_cm = AMD_ENV::cur_env()[src_id].get_available_stream();
    clEnqueueReadBuffer(src_cm, src_mem, CL_TRUE, src_offset, count, (char *) dst + dst_offset, 0, NULL, &event);
    clFlush(src_cm);
    clWaitForEvents(1, &event);
    LOG(INFO) << "OpenCL, sync, D2H, size: " << count;
}
#endif

void AMD_API::async_memcpy(void* dst, int dst_id, const TPtr src, int src_id, \
        size_t count, stream_t& stream, __DtoH) {
    //async_memcpy_with_offset(dst, dst_id, 0, src, src_id, 0, count, stream, __DtoH());
    cl_mem src_mem = src.dmem;
    size_t src_offset = src.offset;

    LOG(INFO)  << __func__<< " D2H dst=" << dst << " dst_id=" << dst_id << " src=" << (void*)src_mem << " src_id=" << src_id << " src_offset=" << src_offset << " count=" << count;

    //cl_mem src_mem = (cl_mem) src;
    cl_command_queue src_cm = AMD_ENV::cur_env()[src_id].get_available_stream(stream);
    clEnqueueReadBuffer(src_cm, src_mem, CL_FALSE, src_offset, count, dst, 0, NULL, NULL);
    clFlush(src_cm);
    LOG(INFO) << "OpenCL, async, D2H, size: " << count;
}
#if 0
void AMD_API::async_memcpy_with_offset(void* dst, int dst_id, size_t dst_offset, const void* src, int src_id, size_t src_offset, \
        size_t count, stream_t& stream, __DtoH) {

    LOG(INFO)  << __func__<< " D2H dst=" << dst << " dst_id=" << dst_id << " dst_office=" << dst_offset << " src=" <<src << " src_id=" << src_id << " src_offset=" << src_offset << " count=" << count;

    cl_mem src_mem = (cl_mem) src;
    cl_command_queue src_cm = AMD_ENV::cur_env()[src_id].get_available_stream(stream);
    clEnqueueReadBuffer(src_cm, src_mem, CL_FALSE, src_offset, count, (char *)dst + dst_offset, 0, NULL, NULL);
    clFlush(src_cm);
    LOG(INFO) << "OpenCL, async, D2H, size: " << count;
}
#endif

void AMD_API::sync_memcpy_p2p(TPtr dst, int dst_dev, const TPtr src, \
        int src_dev, size_t count) {

    cl_mem dst_mem = dst.dmem;
    cl_mem src_mem = src.dmem;

    size_t dst_offset = dst.offset;
    size_t src_offset = src.offset;

    cl_command_queue dst_cm = AMD_ENV::cur_env()[dst_dev].get_available_stream();
    cl_command_queue src_cm = AMD_ENV::cur_env()[src_dev].get_available_stream();

    cl_int err;
    cl_event event;
    void *host_ptr = clEnqueueMapBuffer(src_cm, src_mem, CL_TRUE, CL_MAP_READ, 0, count, 0, NULL, NULL, &err);
    AMD_CHECK(err);
    AMD_CHECK(clEnqueueWriteBuffer(dst_cm, dst_mem, CL_TRUE, 0, count, host_ptr, 0, NULL, NULL));
    AMD_CHECK(clEnqueueUnmapMemObject(src_cm, src_mem, host_ptr, 0, NULL, &event));
    clFlush(src_cm);
    clFlush(dst_cm);
    clWaitForEvents(1, &event);
    LOG(INFO) << "OpenCL, sync, P2P, size: " << count;
}
        
void AMD_API::async_memcpy_p2p(TPtr dst, int dst_dev, const TPtr src, \
        int src_dev, size_t count, stream_t& stream) {

    cl_mem dst_mem = dst.dmem;
    cl_mem src_mem = src.dmem;

    size_t dst_offset = dst.offset;
    size_t src_offset = src.offset;

    cl_command_queue dst_cm = AMD_ENV::cur_env()[dst_dev].get_available_stream(stream);
    cl_command_queue src_cm = AMD_ENV::cur_env()[src_dev].get_available_stream(stream);

    cl_int err;
    cl_event dst_event, src_event, src_event2;
    void *host_ptr = clEnqueueMapBuffer(src_cm, src_mem, CL_FALSE, CL_MAP_READ, 0, count, 0, NULL, &dst_event, &err);
    AMD_CHECK(err);
    AMD_CHECK(clEnqueueWriteBuffer(dst_cm, dst_mem, CL_FALSE, 0, count, host_ptr, 1, &dst_event, &src_event));
    AMD_CHECK(clEnqueueUnmapMemObject(src_cm, src_mem, host_ptr, 1, &src_event2, NULL));
    clFlush(src_cm);
    clFlush(dst_cm);
    LOG(INFO) << "OpenCL, async, P2P, size: " << count;
}

/**
 * \brief device target return currently used device id
 * @return          currently activated device id
 */
int AMD_API::get_device_id(){
    //LOG(INFO) << "get device id = " << current_device_id_index;
    return current_device_id_index;
}

void get_mem_from_ptr(void *ptr, cl_mem *mem){

    std::map<void*, cl_mem>::iterator it;
    it = AMD_API::buffers.find(ptr);
    if(it != AMD_API::buffers.end()){
        *mem = it->second;
    } else {
        *mem = NULL;
    }
}

cl_platform_id AMD_API::get_platform_id(){

    cl_int errNum;
    cl_uint nums;
    cl_platform_id id = NULL;

    errNum = clGetPlatformIDs(0, 0, &nums);
    if (nums <=0) {
          AMD_CHECK_MSG(errNum,"Failed to find any OpenCL platforms");
     }

    cl_platform_id platformIDs[nums];
    errNum = clGetPlatformIDs(nums, &platformIDs[0], &nums);
    if(errNum != CL_SUCCESS || nums <=0)
    {
         AMD_CHECK_MSG(errNum, "Failed to get OpenCL platforms");
    }

    errNum = CL_INVALID_VALUE;
    for (cl_uint i = 0; i < nums; i++) {
        size_t infoSize;
        AMD_CHECK(clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, 0, NULL, &infoSize));
        char *info = (char*) malloc(infoSize);
        AMD_CHECK(clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, infoSize, info, NULL));

        if (strstr(info, "AMD") != NULL) {
            id = platformIDs[i];
            errNum = CL_SUCCESS;
            free(info);
            break;
        }
        free(info);
     }

    AMD_CHECK_MSG(errNum, "There is no AMD Platform");
    return id;
}

/*
void AMD_API::init(){

    if(enable_amd)
        return;

    cl_int errNum;
    cl_uint nums;
    errNum = clGetPlatformIDs(0, 0, &nums);
    if (nums <=0) {
          AMD_CHECK_MSG(errNum,"Failed to find any OpenCL platforms");
     }

    cl_platform_id platformIDs[nums];
    errNum = clGetPlatformIDs(nums, &platformIDs[0], &nums);
    if(errNum != CL_SUCCESS || nums <=0)
    {
         AMD_CHECK_MSG(errNum, "Failed to get OpenCL platforms");
    }


    for (cl_uint i = 0; i < nums; i++) {
        size_t infoSize;
        clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, 0, NULL, &infoSize);
        char *info = (char*) malloc(infoSize);
        clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, infoSize, info, NULL);

        if (strstr(info, "AMD") != NULL) {
            platform_id = platformIDs[i];
            enable_amd = CL_SUCCESS;
            free(info);
            break;
        }
        free(info);
     }

    AMD_CHECK_MSG(enable_amd, "There is no AMD Platform");

    //init devices map
    AMD_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &nums));

    if(nums <= 0)
        AMD_CHECK_MSG(CL_INVALID_VALUE, "There is no AMD GPU");

    device_ids = new cl_device_id[nums];
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, nums, device_ids, &device_nums);
    current_device_id_index = 0;


    //init context, one by one mapping to device.
    contexts = new cl_context[nums];
    const cl_context_properties prop[] = {
         CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id,
         0
    };
    for (cl_uint i = 0; i < nums; i++) {
        contexts[nums] = clCreateContext(prop, 1, &device_ids[i], NULL, NULL, &errNum);
    }
}
*/

//! AMD TargetWrapper
template struct TargetWrapper<AMD, __device_target>;

//! AMD Buffer
INSTANTIATE_BUFFER(AMD);

//! AMD Tensor

INSTANTIATE_TENSOR(AMD, AK_FLOAT, NCHW);
INSTANTIATE_TENSOR(AMD, AK_FLOAT, NHWC);
INSTANTIATE_TENSOR(AMD, AK_FLOAT, HW);
INSTANTIATE_TENSOR(AMD, AK_FLOAT, NHW);
INSTANTIATE_TENSOR(AMD, AK_FLOAT, NW);

INSTANTIATE_TENSOR(AMD, AK_INT8, NCHW);
INSTANTIATE_TENSOR(AMD, AK_INT8, NHWC);
INSTANTIATE_TENSOR(AMD, AK_INT8, HW);
INSTANTIATE_TENSOR(AMD, AK_INT8, NHW);
INSTANTIATE_TENSOR(AMD, AK_INT8, NW);

INSTANTIATE_TENSOR(AMD, AK_HALF, NCHW);
INSTANTIATE_TENSOR(AMD, AK_HALF, NHWC);
INSTANTIATE_TENSOR(AMD, AK_HALF, HW);
INSTANTIATE_TENSOR(AMD, AK_HALF, NHW);
INSTANTIATE_TENSOR(AMD, AK_HALF, NW);
//!
template struct Env<AMD>;

#endif // AMD_GPU

} //namespace saber

} //namespace anakin
