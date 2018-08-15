#include "framework/operators/softmax.h"

namespace anakin {

namespace ops {

#define INSTANCE_SOFTMAX(Ttype, Dtype, Ptype) \
template<> \
void Softmax<Ttype, Dtype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<SoftmaxHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<SoftmaxHelper<Ttype, Dtype, Ptype>*>\
                  (this->_helper)->_param_softmax; \
    impl->_funcs_softmax(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SoftmaxHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Softmax op parameter.";
    auto axis = GET_PARAMETER(int, axis);

    SoftmaxParam<Tensor4d<Ttype, Dtype>> param_softmax(axis);
    _param_softmax = param_softmax;
    return Status::OK();
}

template<>
Status SoftmaxHelper<X86, AK_FLOAT, Precision::FP32>::Init(OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86, AK_FLOAT> >& ins,
        std::vector<Tensor4dPtr<X86, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
template<>
Status SoftmaxHelper<X86, AK_FLOAT, Precision::FP16>::Init(OpContext<X86>& ctx,
                                                           const std::vector<Tensor4dPtr<X86, AK_FLOAT> >& ins,
                                                           std::vector<Tensor4dPtr<X86, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<>
Status SoftmaxHelper<X86, AK_FLOAT, Precision::INT8>::Init(OpContext<X86>& ctx,
                                                           const std::vector<Tensor4dPtr<X86, AK_FLOAT> >& ins,
                                                           std::vector<Tensor4dPtr<X86, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SoftmaxHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                           std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, STATIC, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SoftmaxHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                                 std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    SABER_CHECK(_funcs_softmax.compute_output_shape(ins, outs, _param_softmax));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SOFTMAX(NV, AK_FLOAT, Precision::FP32);
template class SoftmaxHelper<NV, AK_FLOAT, Precision::FP32>;
template class SoftmaxHelper<NV, AK_FLOAT, Precision::FP16>;
template class SoftmaxHelper<NV, AK_FLOAT, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_SOFTMAX(X86, AK_FLOAT, Precision::FP32);
template class SoftmaxHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SOFTMAX(ARM, AK_FLOAT, Precision::FP32);
template <>
Status SoftmaxHelper<ARM, AK_FLOAT, Precision::FP32>::Init(OpContext<ARM> &ctx, \
    const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins, \
    std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_AMD
INSTANCE_SOFTMAX(AMD, AK_FLOAT, Precision::FP32);
template <>
Status SoftmaxHelper<AMD, AK_FLOAT, Precision::FP32>::Init(OpContext<AMD> &ctx, \
    const std::vector<Tensor4dPtr<AMD, AK_FLOAT> >& ins, \
    std::vector<Tensor4dPtr<AMD, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, AMD, AK_FLOAT, Precision::FP32);
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
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("softmax")
#endif
#ifdef USE_AMD
.__alias__<AMD, AK_FLOAT, Precision::FP32>("softmax")
#endif
.num_in(1)
.num_out(1)
.Args<int>("axis", " axis ");

} /* namespace ops */

} /* namespace anakin */


