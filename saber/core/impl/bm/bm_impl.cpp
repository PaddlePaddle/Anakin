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

//TODO: check exception
//static bm_handle_t handle = get_bm_handle();
static bm_handle_t handle;

bm_handle_t BM_API::get_handler() {
    return handle;
}

void BM_API::get_device_count(int &count) {
    BMDNN_CHECK(bm_dev_getcount(&count));
}

void BM_API::set_device(int id){
    //(bm_handle_t &handle, bool bmkernel_used, int id){
    BMDNN_CHECK(bm_dev_request(&handle, 0, id));
}

//TODO: Do we have this functionality?
int BM_API::get_device_id(){
    return 0;
}
        
void BM_API::mem_alloc(void** ptr, size_t n){
    //(bm_handle_t handle, bm_device_mem_t* pmem, unsigned int n)
    bm_device_mem_t mem = bm_mem_from_system(*ptr);
    BMDNN_CHECK(bm_malloc_device_byte(handle, &mem, n));
    //bm_device_mem_t* pmem = (struct bm_mem_desc *)(*ptr);
    //BMDNN_CHECK(bm_malloc_device_byte(handle, pmem, n));
    //bm_device_mem_t mem = bm_mem_from_system(*ptr);
    handle = get_bm_handle();
    bm_device_mem_t *mem = new bm_device_mem_t[1];
    mem = reinterpret_cast<struct bm_mem_desc *>(ptr);
    BMDNN_CHECK(bm_malloc_device_byte(handle, mem, n));
}
        
void BM_API::mem_free(void* ptr){
    //(bm_handle_t handle, bm_device_mem_t mem){
    if(ptr != nullptr){
        bm_free_device(handle, bm_mem_from_system(ptr));
        //bm_device_mem_t* pmem = (struct bm_mem_desc *)(ptr);
        //bm_free_device(handle, *pmem);
        //handle = get_bm_handle();
	//bm_free_device(handle, reinterpret_cast<struct bm_mem_desc>(*ptr));
    }
}
        
void BM_API::mem_set(void* ptr, int value, size_t n){
    //(bm_handle_t handle, const int value, bm_device_mem_t mem){
    BMDNN_CHECK(bm_memset_device(handle, value, bm_mem_from_system(ptr)));
    //bm_device_mem_t* pmem = (struct bm_mem_desc *)(ptr);
    //BMDNN_CHECK(bm_memset_device(handle, value, *pmem));
}

//! target wrapper
template struct TargetWrapper<BM, __device_target>;

//! BM Buffer
template class Buffer<BM>;

//! BM Tensor
INSTANTIATE_TENSOR(BM, AK_BM, NCHW);

template struct Env<BM>;

#endif //USE_BM

} //namespace saber

} //namespace anakin
