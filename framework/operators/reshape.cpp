#include "framework/operators/reshape.h"

namespace anakin {

namespace ops {

#define INSTANCE_RESHAPE(Ttype, Dtype, Ptype) \
template<> \
void Reshape<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<ReshapeHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ReshapeHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_reshape; \
    impl->_funcs_reshape(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ReshapeHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Reshape op parameter.";
    auto dims = GET_PARAMETER(PTuple<int>, dims);

    ReshapeParam<Tensor4d<Ttype, Dtype>> param_reshape(dims.vector());
    _param_reshape = param_reshape;
    return Status::OK();
}


template<typename Ttype, DataType Dtype, Precision Ptype>
Status ReshapeHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                           std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    SABER_CHECK(_funcs_reshape.init(ins, outs, _param_reshape, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}


template<typename Ttype, DataType Dtype, Precision Ptype>
Status ReshapeHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                                 std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    SABER_CHECK(_funcs_reshape.compute_output_shape(ins, outs, _param_reshape));
    outs[0]->set_seq_offset(ins[0]->get_seq_offset());
    return Status::OK();
}


#ifdef USE_CUDA
INSTANCE_RESHAPE(NV, AK_FLOAT, Precision::FP32);
template class ReshapeHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_RESHAPE(X86, AK_FLOAT, Precision::FP32);
template class ReshapeHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_RESHAPE(ARM, AK_FLOAT, Precision::FP32);
template class ReshapeHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Reshape)
.Doc("Reshape operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("reshape")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("reshape")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("reshape")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims of redhape target");

} /* namespace ops */

} /* namespace anakin */


