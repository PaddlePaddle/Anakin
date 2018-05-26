#include "framework/operators/reshape.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Reshape<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<ReshapeHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<ReshapeHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_reshape;
    impl->_funcs_reshape(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
ReshapeHelper<Ttype, Dtype, Ptype>::~ReshapeHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ReshapeHelper<Ttype, Dtype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing Reshape op parameter.";
    auto dims = GET_PARAMETER(PTuple<int>, dims);

    ReshapeParam<Tensor4d<Ttype, Dtype>> param_reshape(dims.vector());
    _param_reshape = param_reshape;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ReshapeHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_reshape.init(ins, outs, _param_reshape, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ReshapeHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_reshape.compute_output_shape(ins, outs, _param_reshape));
    outs[0]->set_seq_offset(ins[0]->get_seq_offset());
    return Status::OK();
}

#ifdef USE_CUDA
template class ReshapeHelper<NV, AK_FLOAT, Precision::FP32>;
template class ReshapeHelper<NV, AK_FLOAT, Precision::FP16>;
template class ReshapeHelper<NV, AK_FLOAT, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class ReshapeHelper<ARM, AK_FLOAT, Precision::FP32>;
template class ReshapeHelper<ARM, AK_FLOAT, Precision::FP16>;
template class ReshapeHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
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
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims of redhape target");

} /* namespace ops */

} /* namespace anakin */


