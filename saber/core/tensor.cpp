#include "tensor.h"

#ifdef USE_BM

#include "bmlib_runtime.h"
#include "bmdnn_api.h"
#include "bmlib_utils.h"

template<>
template<>
SaberStatus Tensor<BM, AK_BM, NCHW>::copy_from<X86, AK_FLOAT, NCHW>(const Tensor<X86, AK_FLOAT, NCHW>& tensor) {
    //auto* device_data_ptr = mutable_data();
    //BMDNN_CHECK(bm_memcpy_s2d(API::get_handler(), *device_data_ptr, bm_mem_from_system(tensor.data())));
    return SaberSuccess;
}

template<>
template<>
SaberStatus Tensor<X86, AK_FLOAT, NCHW>::copy_from<BM, AK_BM, NCHW>(const Tensor<BM, AK_BM, NCHW>& tensor) {
    return SaberSuccess;
}

#endif

