#include "framework/operators/flatten.h"

namespace anakin {

namespace ops {

#define INSTANCE_FLATTEN(Ttype, Dtype, Ptype) \
template<> \
void Flatten<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<FlattenHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<FlattenHelper<Ttype, Dtype, Ptype>*> \
                  (this->_helper)->_param_flatten; \
    impl->_funcs_flatten(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status FlattenHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Flatten op parameter.";
    auto start_axis = GET_PARAMETER(int, start_axis);
    auto end_axis = GET_PARAMETER(int, end_axis);

    saber::FlattenParam<Tensor4d<Ttype, Dtype>> flatten_param;
    _param_flatten = flatten_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status FlattenHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                           std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    SABER_CHECK(_funcs_flatten.init(ins, outs, _param_flatten, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
template<typename Ttype, DataType Dtype, Precision Ptype>
Status FlattenHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                                 std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    SABER_CHECK(_funcs_flatten.compute_output_shape(ins, outs, _param_flatten));
    return Status::OK();
}
#ifdef USE_CUDA
INSTANCE_FLATTEN(NV, AK_FLOAT, Precision::FP32);
template class FlattenHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Flatten, FlattenHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_FLATTEN(X86, AK_FLOAT, Precision::FP32);
template class FlattenHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Flatten, FlattenHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_FLATTEN(ARM, AK_FLOAT, Precision::FP32);
template class FlattenHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Flatten, FlattenHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Flatten)
.Doc("Flatten operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("flatten")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("flatten")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("flatten")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


