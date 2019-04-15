#include "framework/operators/roi_pool.h"

namespace anakin {

namespace ops {

#define INSTANCE_ROI_POOL(Ttype, Ptype) \
template<> \
void RoiPool<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<RoiPoolHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<RoiPoolHelper<Ttype, Ptype>*>(this->_helper)->_param_roi_pool; \
    impl->_funcs_roi_pool(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
RoiPoolHelper<Ttype, Ptype>::~RoiPoolHelper() {
}

template<typename Ttype, Precision Ptype>
Status RoiPoolHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing RoiPool op parameter.";
    auto pooled_height = GET_PARAMETER(int, pooled_h);
    auto pooled_width = GET_PARAMETER(int, pooled_w);
    auto spatial_scale = GET_PARAMETER(float, spatial_scale);
    RoiPoolParam<Ttype> param_roi_pool(pooled_height, pooled_width, spatial_scale);
    _param_roi_pool = param_roi_pool;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status RoiPoolHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
                                          const std::vector<Tensor4dPtr<Ttype> >& ins,
                                          std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_roi_pool.init(ins, outs, _param_roi_pool, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status RoiPoolHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                                                std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_roi_pool.compute_output_shape(ins, outs, _param_roi_pool));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ROI_POOL(NV, Precision::FP32);
template<>
Status RoiPoolHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx, \
                    const std::vector< Tensor4dPtr<NV> > & ins, std::vector< Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_roi_pool.init(ins, outs, _param_roi_pool, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(RoiPool, RoiPoolHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ROI_POOL(X86, Precision::FP32);
INSTANCE_ROI_POOL(X86, Precision::FP16);
INSTANCE_ROI_POOL(X86, Precision::INT8);
template class RoiPoolHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(RoiPool, RoiPoolHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ROI_POOL(ARM, Precision::FP32);
template class RoiPoolHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(RoiPool, RoiPoolHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_ROI_POOL(AMD, Precision::FP32);
template class RoiPoolHelper<AMD, Precision::FP32>;
template class RoiPoolHelper<AMD, Precision::FP16>;
template class RoiPoolHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(RoiPool, RoiPoolHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(RoiPool)
        .Doc("RoiPool operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("roi_pool")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("roi_pool")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("roi_pool")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("roi_pool")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of RoiPool ")
.Args<int>("pooled_h", "roi pool height")
.Args<int>("pooled_w", "roi pool width")
.Args<float>("spatial_scale", "roi pool spatial_scale");

} /* namespace ops */

} /* namespace anakin */



