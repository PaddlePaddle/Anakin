#include "core/tensor.h"
#include "env.h"

#include "bmlib_runtime.h"
#include "bmdnn_api.h"
#include "bmlib_utils.h"

#ifdef USE_BM
const char* bmdnn_get_errorstring(bm_status_t error) {
    switch (error) {
        case BM_SUCCESS:
            return "BM API call correct";
        case BM_ERR_FAILURE:
            return "BM API fail to return";
        case BM_ERR_TIMEOUT:
            return "BM API time out";
        case BM_ERR_PARAM:
            return "BM API invalid parameter";
        case BM_ERR_NOMEM:
            return "BM API insufficient memory";
        case BM_ERR_DATA:
            return "BM API invalid data";
        case BM_ERR_BUSY:
            return "BM device is busy";
        case BM_NOT_SUPPORTED:
            return "BM unsupported operate";
    }
    return "Unknown bmdnn status";
}
#endif

namespace anakin{

namespace saber{

#ifdef USE_BM

typedef TargetWrapper<BM, __device_target> BM_API;

// Init handle only once in the lifetime
static bm_handle_t handle;
static bm_status_t init_handle{bmdnn_init(&handle)};

bm_handle_t BM_API::get_handle() {
    /*bm_handle_t handle;
    int ret = 0;

    ret = bm_dev_request(&handle, 0, devid);
    CHECK_NE(ret, 0) << "request BM device failed: " << devid;
    */
    return handle;
};

void BM_API::get_device_count(int &count) {
    BMDNN_CHECK(bm_dev_getcount(&count));
}

void BM_API::set_device(int id){
    //(bm_handle_t &handle, bool bmkernel_used, int id){
    //BMDNN_CHECK(bm_dev_request(&handle, 0, id));
}

//TODO: Do we have this functionality?
int BM_API::get_device_id(){
    return 0;
}
        
void BM_API::mem_alloc(void** ptr, size_t n){
    //handle = BM_API::get_handle();
    /* bm_device_mem_t *mem = reinterpret_cast<struct bm_mem_desc *>(*ptr); */
    bm_device_mem_t *mem = new bm_device_mem_t();
    BMDNN_CHECK(bm_malloc_device_byte(handle, mem, n));
    *ptr = mem;
}
        
void BM_API::mem_free(void* ptr){
    if(ptr != nullptr){
        //handle = BM_API::get_handle();
        bm_free_device(handle, *(struct bm_mem_desc*)(ptr));
        delete ptr;
    }
}
        
void BM_API::mem_set(void* ptr, int value, size_t n){
    //(bm_handle_t handle, const int value, bm_device_mem_t mem){
    BMDNN_CHECK(bm_memset_device(handle, value, bm_mem_from_system(ptr)));
    //bm_device_mem_t* pmem = (struct bm_mem_desc *)(ptr);
    //BMDNN_CHECK(bm_memset_device(handle, value, *pmem));
}

void BM_API::sync_memcpy(void* dst, size_t dst_offset, int dst_id, \
        const void* src, size_t src_offset, int src_id, \
        size_t count, __DtoD) {
    //handle = BM_API::get_handle(); 
    //BMDNN_CHECK(bm_memcpy_d2d(handle, bm_mem_from_device(dst), dst_id, bm_mem_from_device(src), src_id, count));
    BMDNN_CHECK(bm_memcpy_d2d(handle, *(bm_device_mem_t *)(dst), dst_id, *(bm_device_mem_t *)(src), src_id, count));
    LOG(INFO) << "BM sync_memcpy: device to device, finished";
};

void BM_API::sync_memcpy(void* dst, size_t dst_offset, int dst_id, \
        const void* src, size_t src_offset, int src_id, \
        size_t count, __HtoD) {
    //handle = BM_API::get_handle(); 
    BMDNN_CHECK(bm_memcpy_s2d(handle, *(bm_device_mem_t *)(dst), bm_mem_from_system(src)));

    #ifdef DEBUG
    for(int i=0; i<10; i++)
	    LOG(INFO) << "HtoD src: " << *((float *)(src)+i);
    #endif
    
    LOG(INFO) << "BM sync_memcpy: host to device, finished";
};

void BM_API::sync_memcpy(void* dst, size_t dst_offset, int dst_id, \
        const void* src, size_t src_offset, int src_id, \
        size_t count, __DtoH) {
    //handle = BM_API::get_handle(); 
    BMDNN_CHECK(bm_memcpy_d2s(handle, bm_mem_from_system(dst), *(bm_device_mem_t *)(src)));

    #ifdef DEBUG
    for(int i=0; i<10; i++)
        LOG(INFO) << "DtoH dst: " << *((float *)(dst)+i);
    #endif

    LOG(INFO) << "BM sync_memcpy: device to host, finished";
};

void BM_API::sync_memcpy_p2p(void* dst, size_t dst_offset, int dst_id, \
        const void* src, size_t src_offset, int src_id, \
        size_t count) { 

    LOG(ERROR) << "BM sync_memcpy_p2p: temporarily no used";
};

//! BM TargetWrapper
template struct TargetWrapper<BM, __device_target>;

//! BM Buffer
template class Buffer<BM>;

//! BM Tensor
template class Tensor<BM>;

//! BM Env
template struct Env<BM>;

#endif //USE_BM

} //namespace saber

} //namespace anakin
