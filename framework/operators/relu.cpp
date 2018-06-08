#include "framework/operators/relu.h"

namespace anakin {

namespace ops {

//#ifdef USE_CUDA
//template<>
//void ReLU<NV, AK_FLOAT, Precision::FP32>::operator()(
//    OpContext<NV>& ctx,
//    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
//    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
//    auto* impl =
//        static_cast<ReLUHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
//    auto& param = impl->_param_relu;
//    impl->_funcs_relu(ins, outs, param, ctx);
//}
//#endif

/// TODO ... specialization other type of operator
#define INSTANCE_RELU(Ttype, Dtype, Ptype) \
template<> \
void ReLU<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,\
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<ReLUHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_relu; \
    impl->_funcs_relu(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
ReLUHelper<Ttype, Dtype, Ptype>::~ReLUHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ReLUHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing ReLU op parameter.";

    // get relu param
    auto alpha = GET_PARAMETER(float, alpha);
    ActivationParam<Tensor4d<Ttype, Dtype>> active_param(Active_relu);//, alpha); // TEMP
    _param_relu = active_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ReLUHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_relu.init(ins, outs, _param_relu, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ReLUHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_relu.compute_output_shape(ins, outs, _param_relu));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_RELU(NV, AK_FLOAT, Precision::FP32);
template class ReLUHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReLU, ReLUHelper, NV, AK_FLOAT, Precision::FP32);
template class ReLUHelper<NV, AK_FLOAT, Precision::FP16>;
template class ReLUHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE

#ifdef ANAKIN_TYPE_FP32
INSTANCE_RELU(ARM, AK_FLOAT, Precision::FP32);
template class ReLUHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReLU, ReLUHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef ANAKIN_TYPE_FP16
template class ReLUHelper<ARM, AK_FLOAT, Precision::FP16>;
#endif
#ifdef ANAKIN_TYPE_INT8
template class ReLUHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

#endif//arm

//! register op
ANAKIN_REGISTER_OP(ReLU)
.Doc("ReLU operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("Relu")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("Relu")
#endif
.num_in(1)
.num_out(1)
.Args<float>("alpha", " alpha for relu");

} /* namespace ops */

} /* namespace anakin */


