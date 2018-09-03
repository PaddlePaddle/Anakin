#include "framework/operators/resize.h"

namespace anakin {

namespace ops {

#define INSTANCE_RESIZE(Ttype, Dtype, Ptype) \
template<> \
void Resize<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,\
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<ResizeHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_resize; \
    impl->_funcs_resize(ins, outs, param, ctx); \
}


template<typename Ttype, DataType Dtype, Precision Ptype>
Status ResizeHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Resize op parameter.";

    // get resize param
    auto width_scale = GET_PARAMETER(float, width_scale);
    auto height_scale = GET_PARAMETER(float, height_scale);
    
    ResizeParam<Tensor4d<Ttype, Dtype>> resize_param(height_scale, width_scale);
    _param_resize = resize_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ResizeHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                        std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    SABER_CHECK(_funcs_resize.init(ins, outs, _param_resize, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ResizeHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                              std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    SABER_CHECK(_funcs_resize.compute_output_shape(ins, outs, _param_resize));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_RESIZE(NV, AK_FLOAT, Precision::FP32);
template <>
Status ResizeHelper<NV, AK_FLOAT, Precision::FP32>::Init(OpContext<NV> &ctx,
                                                        const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
                                                        std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
     SABER_CHECK(_funcs_resize.init(ins, outs, _param_resize, SPECIFY, SABER_IMPL, ctx));
     return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Resize, ResizeHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_RESIZE(X86, AK_FLOAT, Precision::FP32);
template class ResizeHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Resize, ResizeHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_RESIZE(ARM, AK_FLOAT, Precision::FP32);
template class ResizeHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Resize, ResizeHelper, ARM, AK_FLOAT, Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(Resize)
.Doc("Resize operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("Resize")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("Resize")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("Resize")
#endif
.num_in(1)
.num_out(1)
.Args<float>("height_scale", " height scale for resize")
.Args<float>("width_scale", " width scale for resize");

} /* namespace ops */

} /* namespace anakin */


