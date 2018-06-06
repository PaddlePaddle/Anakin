#include "framework/operators/softmax.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Softmax<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl = static_cast<SoftmaxHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<SoftmaxHelper<NV, AK_FLOAT, Precision::FP32>*>
                  (this->_helper)->_param_softmax;
    impl->_funcs_softmax(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator
#ifdef USE_ARM_PLACE
template<>
void Softmax<ARM, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<ARM>& ctx,
    const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    auto* impl = static_cast<SoftmaxHelper<ARM, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<SoftmaxHelper<ARM, AK_FLOAT, Precision::FP32>*>
                  (this->_helper)->_param_softmax;
    impl->_funcs_softmax(ins, outs, param, ctx);
}
#endif

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
SoftmaxHelper<Ttype, Dtype, Ptype>::~SoftmaxHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SoftmaxHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Softmax op parameter.";
    auto axis = GET_PARAMETER(int, axis);

    SoftmaxParam<Tensor4d<Ttype, Dtype>> param_softmax(axis);
    _param_softmax = param_softmax;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SoftmaxHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, STATIC, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SoftmaxHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_softmax.compute_output_shape(ins, outs, _param_softmax));
    return Status::OK();
}

#ifdef USE_CUDA
template class SoftmaxHelper<NV, AK_FLOAT, Precision::FP32>;
template class SoftmaxHelper<NV, AK_FLOAT, Precision::FP16>;
template class SoftmaxHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class SoftmaxHelper<ARM, AK_FLOAT, Precision::FP32>;
#endif
#ifdef ANAKIN_TYPE_FP16
template class SoftmaxHelper<ARM, AK_FLOAT, Precision::FP16>;
#endif
#ifdef ANAKIN_TYPE_INT8
template class SoftmaxHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, ARM, AK_FLOAT, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Softmax)
.Doc("Softmax operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("softmax")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("softmax")
#endif
.num_in(1)
.num_out(1)
.Args<int>("axis", " axis ");

} /* namespace ops */

} /* namespace anakin */


