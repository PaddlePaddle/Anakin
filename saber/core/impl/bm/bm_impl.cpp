#include "core/tensor.h"
#include "core/common.h"
#include "core/data_traits.h"
#include "env.h"

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


namespace anakin {

namespace saber {



typedef TargetWrapper<BM, __device_target> BM_API;


// Init handle only once in the lifetime
static bm_handle_t handle;
static bm_status_t init_handle{bmlib_kernel_init(&handle)};

bm_handle_t BM_API::get_handle() {
    return handle;
};

void BM_API::get_device_count(int& count) {
    BM_CHECK(bm_dev_getcount(&count));
}

void BM_API::set_device(int id) {
    //(bm_handle_t &handle, bool bmkernel_used, int id){
    //BM_CHECK(bm_dev_request(&handle, 0, id));
}

//TODO: Do we have this functionality?
int BM_API::get_device_id() {
    return 0;
}

void BM_API::mem_alloc(TPtr* ptr, size_t n) {
    /* bm_device_mem_t *mem = reinterpret_cast<struct bm_mem_desc *>(*ptr); */
    //    bm_device_mem_t *mem = new bm_device_mem_t();
    bm_device_mem_t mem;
    BM_CHECK(bm_malloc_device_byte(handle, &mem, n));
    *ptr = TPtr(mem);
}

void BM_API::mem_free(TPtr ptr) {
    if (bm_mem_get_type(ptr) == BM_MEM_TYPE_SYSTEM) {
        bm_free_device(handle, ptr);
        //        delete ptr;
    }
}

void BM_API::mem_set(TPtr ptr, int value, size_t n) {
    //(bm_handle_t handle, const int value, bm_device_mem_t mem){
    BM_CHECK(bm_memset_device(handle, value, ptr));
    //bm_device_mem_t* pmem = (struct bm_mem_desc *)(ptr);
    //BM_CHECK(bm_memset_device(handle, value, *pmem));
}

void BM_API::sync_memcpy(TPtr dst, size_t dst_offset, int dst_id, \
        const TPtr src, size_t src_offset, int src_id, \
        size_t count, __DtoD) {
    if(count==0)
        return;
    //BM_CHECK(bm_memcpy_d2d(handle, bm_mem_from_device(dst), dst_id, bm_mem_from_device(src), src_id, count));
    BM_CHECK(bm_memcpy_d2d(handle, dst, dst_offset, src, src_offset, count));
};

void BM_API::sync_memcpy(TPtr dst, size_t dst_offset, int dst_id, \
        const void* src, size_t src_offset, int src_id, \
        size_t count, __HtoD) {
    if(count==0)
        return;
    BM_CHECK(bm_memcpy_s2d(handle, dst+dst_offset, bm_mem_from_system(const_cast<void*>(src)+src_offset)));

#ifdef DEBUG

    for (int i = 0; i < 10; i++) {
        LOG(INFO) << "HtoD src: " << *((float*)(src) + i);
    }

#endif
};

void BM_API::sync_memcpy(void* dst, size_t dst_offset, int dst_id, \
        const TPtr src, size_t src_offset, int src_id, \
        size_t count, __DtoH) {
    if(count==0)
        return;
//    LOG(INFO)<<"host ptr = "<<(dst)<<",dst_offset = "<<dst_offset<<", dev ptr = "<<(src)<<",dev offset = "<<src_offset;
    BM_CHECK(bm_memcpy_d2s(handle, bm_mem_from_system(dst+dst_offset), src+src_offset));

#ifdef DEBUG

    for (int i = 0; i < 10; i++) {
        LOG(INFO) << "DtoH dst: " << *((float*)(dst) + i);
    }

#endif

};

void BM_API::sync_memcpy_p2p(TPtr dst, size_t dst_offset, int dst_id, \
        const TPtr src, size_t src_offset, int src_id, \
        size_t count) {
    if(count==0)
        return;
    LOG(FATAL) << "BM sync_memcpy_p2p: temporarily no used";
};

//! BM TargetWrapper
template struct TargetWrapper<BM, __device_target>;

//! BM Buffer
template class Buffer<BM>;

//! BM Tensor



/**
 * \brief Constructor with allocated data ptr and entire memory shape. only for BM
*/
template <>
template <typename TargetType_t>
Tensor<BM>::Tensor(typename DataTraitBase<TargetType_t>::PtrDtype   data_ptr, TargetType_t target, int id, Shape shape,DataType type = AK_FLOAT) {

    _shape = shape;
    _valid_shape = shape;
    _offset = Shape::zero(shape);
    _dtype = type;
    _type_len = type_length(type);
    std::shared_ptr<Buffer<TargetType_t>> buf_from_date = \
                                       std::make_shared<Buffer<TargetType_t>>(&bm_mem_from_system(const_cast<void*>(data_ptr)),
                                               shape.count() * _type_len, id);

    BufferMemShare(_buf, buf_from_date);
    _is_shared = true;
    _is_subbuf = false;
}
template class Tensor<BM>;


template class Env<BM>;



} //namespace saber

} //namespace anakin
