#include "framework/operators/ps_roi_pooling.h"

namespace anakin {

namespace ops {

#define INSTANCE_PSROIPOOLING(Ttype, Ptype) \
template<> \
void PsRoiPooling<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
        std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<PsRoiPoolingHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<PsRoiPoolingHelper<Ttype, Ptype>*>\
                  (this->_helper)->_param_ps_roi_pooling; \
    impl->_funcs_ps_roi_pooling(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status PsRoiPoolingHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing PsRoiPooling op parameter.";

    auto pooled_width = GET_PARAMETER(int, pooled_width);
    auto pooled_height = GET_PARAMETER(int, pooled_height);
    auto crop_width = GET_PARAMETER(int, crop_width);
    auto crop_height = GET_PARAMETER(int, crop_height);
    auto global_pooling = GET_PARAMETER_WITH_DEFAULT(bool, global_pooling, true);
    auto extra_value = GET_PARAMETER_WITH_DEFAULT(float, extra_value, 0);
    auto method = GET_PARAMETER_WITH_DEFAULT(int, method, 0);
    
    auto spatial_scale = GET_PARAMETER(float, spatial_scale);

    PsRoiPoolParam<Ttype> ps_roi_pooling_param(pooled_height, 
      pooled_width, crop_height, crop_width, method, extra_value, 
      global_pooling,spatial_scale);

    _param_ps_roi_pooling = ps_roi_pooling_param;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PsRoiPoolingHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype>> &ins,
                           std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_ps_roi_pooling.init(ins, outs, _param_ps_roi_pooling, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PsRoiPoolingHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                                 std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_ps_roi_pooling.compute_output_shape(ins, outs, _param_ps_roi_pooling));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_PSROIPOOLING(NV, Precision::FP32);
template <>
Status PsRoiPoolingHelper<NV, Precision::FP32>::Init(OpContext<NV> &ctx, \
    const std::vector<Tensor4dPtr<NV> >& ins, \
    std::vector<Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_ps_roi_pooling.init(ins, outs, _param_ps_roi_pooling, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(PsRoiPooling, PsRoiPoolingHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_PSROIPOOLING(ARM, Precision::FP32);
template class PsRoiPoolingHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PsRoiPooling, PsRoiPoolingHelper, ARM, Precision::FP32);
#endif  //arm

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_PSROIPOOLING(X86, Precision::FP32);
template class PsRoiPoolingHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PsRoiPooling, PsRoiPoolingHelper, X86, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_PSROIPOOLING(AMD, Precision::FP32);
template class PsRoiPoolingHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PsRoiPooling, PsRoiPoolingHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(PsRoiPooling)
.Doc("PsRoiPooling operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("PsRoiPooling")
.__alias__<NV, Precision::FP32>("pool")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("PsRoiPooling")
.__alias__<ARM, Precision::FP32>("pool")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("PsRoiPooling")
.__alias__<X86, Precision::FP32>("pool")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("PsRoiPooling")
.__alias__<AMD, Precision::FP32>("pool")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("method", "PsRoiPooling type to be applied (MAX, SUM, AVG).")
.Args<bool>("cmp_out_shape_floor_as_conv cmp_out_shape_floor_as_conv of PsRoiPooling for adu novel approach")
.Args<bool>("global_PsRoiPooling", "whether execute global PsRoiPooling on input")
.Args<PTuple<int>>("pool_size", " kernel size for PsRoiPooling (x, y) or (x, y, z).")
.Args<PTuple<int>>("strides",  "stride for PsRoiPooling (x, y)  or  (x, y, z).")
.Args<PTuple<int>>("padding", "pad for PsRoiPooling: (x, y) or (x, y, z).");

} /* namespace ops */

} /* namespace anakin */


