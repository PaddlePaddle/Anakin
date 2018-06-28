#include "framework/operators/dense.h"

namespace anakin {

namespace ops {

#define INSTANCE_DENSE(Ttype, Dtype, Ptype) \
template<> \
void Dense<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<DenseHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<DenseHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_dense; \
    LOG(ERROR) << "run fc"; \
    SABER_CHECK(impl->_funcs_dense(ins, outs, param, ctx)); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status DenseHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Dense op parameter.";
    auto axis = GET_PARAMETER(int, axis);
    auto out_dim = GET_PARAMETER(int, out_dim);
    auto bias_term = GET_PARAMETER(bool, bias_term);

	using pblock_type = PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>;
    auto weights = GET_PARAMETER(pblock_type, weight_1);

        if (bias_term) {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
        saber::FcParam<Tensor4d<Ttype, Dtype>> fc_param(&(weights.d_tensor()), &(bias.d_tensor()), out_dim,
                                            axis);
        _param_dense = fc_param;
    } else {
        Tensor4d<Ttype, Dtype>* bias = nullptr;
        saber::FcParam<Tensor4d<Ttype, Dtype>> fc_param(&(weights.d_tensor()), bias, out_dim, axis);
        _param_dense = fc_param;
    }
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status DenseHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_dense.init(ins, outs, _param_dense, STATIC, VENDER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status DenseHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_dense.compute_output_shape(ins, outs, _param_dense));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_DENSE(NV, AK_FLOAT, Precision::FP32);
template class DenseHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Dense, DenseHelper, NV, AK_FLOAT, Precision::FP32);
template class DenseHelper<NV, AK_FLOAT, Precision::FP16>;
template class DenseHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
INSTANCE_DENSE(ARM, AK_FLOAT, Precision::FP32);
template<>
Status DenseHelper<ARM, AK_FLOAT, Precision::FP32>::Init(OpContext<ARM> &ctx,\
        const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins, \
                std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_dense.init(ins, outs, _param_dense, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Dense, DenseHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_DENSE(X86, AK_FLOAT, Precision::FP32);
template class DenseHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Dense, DenseHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_AMD
INSTANCE_DENSE(AMD, AK_FLOAT, Precision::FP32);
template<>
Status DenseHelper<AMD, AK_FLOAT, Precision::FP32>::Init(OpContext<AMD> &ctx,\
        const std::vector<Tensor4dPtr<AMD, AK_FLOAT> >& ins, \
                std::vector<Tensor4dPtr<AMD, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_dense.init(ins, outs, _param_dense, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Dense, DenseHelper, AMD, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Dense)
.Doc("Dense operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("fullconnect")
.__alias__<NV, AK_FLOAT, Precision::FP32>("fc")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("fullconnect")
.__alias__<ARM, AK_FLOAT, Precision::FP32>("fc")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("fullconnect")
.__alias__<X86, AK_FLOAT, Precision::FP32>("fc")
#endif
#ifdef USE_AMD
.__alias__<AMD, AK_FLOAT, Precision::FP32>("fullconnect")
.__alias__<AMD, AK_FLOAT, Precision::FP32>("fc")
#endif
.num_in(1)
.num_out(1)
.Args<int>("axis", " axis to compute ")
.Args<int>("out_dim", " out dim ")
.Args<bool>("bias_term", " whether fc weights have bias");

} /* namespace ops */

} /* namespace anakin */


