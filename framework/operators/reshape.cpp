#include "framework/operators/reshape.h"

namespace anakin {

namespace ops {

#define INSTANCE_RESHAPE(Ttype, Ptype) \
template<> \
void Reshape<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<ReshapeHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ReshapeHelper<Ttype, Ptype>*>(this->_helper)->_param_reshape; \
    impl->_funcs_reshape(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status ReshapeHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Reshape op parameter.";
    auto dims = GET_PARAMETER(PTuple<int>, dims);

    ReshapeParam<Tensor4d<Ttype>> param_reshape(dims.vector());
    _param_reshape = param_reshape;
    return Status::OK();
}


template<typename Ttype, Precision Ptype>
Status ReshapeHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype>> &ins,
                           std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_reshape.init(ins, outs, _param_reshape, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}


template<typename Ttype, Precision Ptype>
Status ReshapeHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                                 std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_reshape.compute_output_shape(ins, outs, _param_reshape));
    outs[0]->set_seq_offset(ins[0]->get_seq_offset());
    return Status::OK();
}


#ifdef USE_CUDA
INSTANCE_RESHAPE(NV, Precision::FP32);
template class ReshapeHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, NV, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_RESHAPE(X86, Precision::FP32);
template class ReshapeHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_RESHAPE(ARM, Precision::FP32);
template class ReshapeHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Reshape)
.Doc("Reshape operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("reshape")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("reshape")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("reshape")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims of redhape target");

} /* namespace ops */

} /* namespace anakin */


