#include "framework/operators/relu.h"

namespace anakin {

namespace ops {

#define INSTANCE_RELU(Ttype, Ptype) \
template<> \
void ReLU<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins,\
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<ReLUHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_relu; \
    impl->_funcs_relu(ins, outs, param, ctx); \
}


template<typename Ttype, Precision Ptype>
Status ReLUHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing ReLU op parameter.";

    // get relu param
    auto alpha = GET_PARAMETER(float, alpha);
    ActivationParam<Ttype> active_param(Active_relu);//, alpha); // TEMP
    _param_relu = active_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReLUHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
                                      const std::vector<Tensor4dPtr<Ttype>>& ins,
                                      std::vector<Tensor4dPtr<Ttype>>& outs) {
    SABER_CHECK(_funcs_relu.init(ins, outs, _param_relu, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReLUHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>>& ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    SABER_CHECK(_funcs_relu.compute_output_shape(ins, outs, _param_relu));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_RELU(NV, Precision::FP32);
template <>
Status ReLUHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_relu.init(ins, outs, _param_relu, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(ReLU, ReLUHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_RELU(X86, Precision::FP32);
template class ReLUHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReLU, ReLUHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_RELU(ARM, Precision::FP32);
template class ReLUHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReLU, ReLUHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_RELU(AMD, Precision::FP32);
template class ReLUHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReLU, ReLUHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(ReLU)
.Doc("ReLU operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("Relu")
#endif

#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("Relu")
#endif

#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("Relu")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("Relu")
#endif
.num_in(1)
.num_out(1)
.Args<float>("alpha", " alpha for relu");

} /* namespace ops */

} /* namespace anakin */


