#include "framework/operators/fusion_ops/dense_dense.h"

namespace anakin {

namespace ops {

#define INSTANCE_DENSEDENSE(Ttype, Ptype) \
template<> \
void DenseDense<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
        std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<DenseDenseHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<DenseDenseHelper<Ttype, Ptype>*>(this->_helper)->_param_dense_dense; \
    SABER_CHECK(impl->_funcs_dense_dense(ins, outs, param, ctx)); \
}

template<typename Ttype, Precision Ptype>
Status DenseDenseHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing DenseDense op parameter.";
    auto axis_0 = GET_PARAMETER(int, axis);
    auto out_dim_0 = GET_PARAMETER_WITH_DEFAULT(int, out_dim,0);
    auto bias_term_0 = GET_PARAMETER(bool, bias_term);

    //now we only support 2 fc fusion
    auto axis_1 = GET_PARAMETER(int, dense_1_axis);
    auto out_dim_1 = GET_PARAMETER_WITH_DEFAULT(int, dense_1_out_dim,0);
    auto bias_term_1 = GET_PARAMETER(bool, dense_1_bias_term);    

    using pblock_type = PBlock<Ttype>;
    auto weights_0 = GET_PARAMETER(pblock_type, weight_1);

    auto weights_1 = GET_PARAMETER(pblock_type, dense_1_weight_1);

    auto weights_dtype = weights_0.h_tensor().get_dtype();

    bool is_transed_0 = CHECK_PARAMETER(is_weights_transposed); 
    bool is_transed_1 = CHECK_PARAMETER(dense_0_is_weights_transposed);

    pblock_type bias_0;
    pblock_type bias_1;
    if (bias_term_0){
        bias_0 = GET_PARAMETER(pblock_type, weight_2);
        if (bias_term_1) {
            bias_1 = GET_PARAMETER(pblock_type, dense_1_weight_2);
        }
    }

    if (weights_dtype == AK_FLOAT) {
        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                WeightsFusion<float, Ttype>::update_dense_weights, 
                weights_0, bias_0, bias_term_0, out_dim_0, is_transed_0,
                weights_1, bias_1, bias_term_1, out_dim_1, is_transed_1);
    } else {
        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                WeightsFusion<char, Ttype>::update_dense_weights, 
                weights_0, bias_0, bias_term_0, out_dim_0, is_transed_0,
                weights_1, bias_1, bias_term_1, out_dim_1, is_transed_1);
    }

    if (bias_term_0 || bias_term_1) {
        saber::FcParam<Ttype> fc_param(&(weights_0.d_tensor()), &(bias_1.d_tensor()), out_dim_1,
                                            axis_1);
        _param_dense_dense = fc_param;
    } else {
        Tensor4d<Ttype>* bias = nullptr;
        saber::FcParam<Ttype> fc_param(&(weights_0.d_tensor()), bias, out_dim_1, axis_1);
        _param_dense_dense = fc_param;
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DenseDenseHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_dense_dense.init(ins, outs, _param_dense_dense, STATIC, SABER_IMPL, ctx));
    return Status::OK();
}
#ifdef USE_CUDA
template<>
Status DenseDenseHelper<NV, Precision::INT8>::Init(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_dense_dense.init(ins, outs, _param_dense_dense, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
#endif
template<>
Status DenseDenseHelper<X86, Precision::FP32>::Init(OpContext<X86>& ctx,
                                       const std::vector<Tensor4dPtr<X86> >& ins,
                                       std::vector<Tensor4dPtr<X86> >& outs) {
    SABER_CHECK(_funcs_dense_dense.init(ins, outs, _param_dense_dense, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}
template<>
Status DenseDenseHelper<X86, Precision::FP16>::Init(OpContext<X86>& ctx,
                                               const std::vector<Tensor4dPtr<X86> >& ins,
                                               std::vector<Tensor4dPtr<X86> >& outs) {
    SABER_CHECK(_funcs_dense_dense.init(ins, outs, _param_dense_dense, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}
#ifndef USE_SGX
template<>
Status DenseDenseHelper<X86, Precision::INT8>::Init(OpContext<X86>& ctx,
                                               const std::vector<Tensor4dPtr<X86> >& ins,
                                               std::vector<Tensor4dPtr<X86> >& outs) {
    SABER_CHECK(_funcs_dense_dense.init(ins, outs, _param_dense_dense, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}
#endif
template<typename Ttype, Precision Ptype>
Status DenseDenseHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_dense_dense.compute_output_shape(ins, outs, _param_dense_dense));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_DENSEDENSE(NV, Precision::FP32);
INSTANCE_DENSEDENSE(NV, Precision::INT8);
template class DenseDenseHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(DenseDense, DenseDenseHelper, NV, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(DenseDense, DenseDenseHelper, NV, Precision::INT8);
template class DenseDenseHelper<NV, Precision::FP16>;
template class DenseDenseHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
INSTANCE_DENSEDENSE(ARM, Precision::FP32);
INSTANCE_DENSEDENSE(ARM, Precision::INT8);
template<>
Status DenseDenseHelper<ARM, Precision::FP32>::Init(OpContext<ARM> &ctx,\
        const std::vector<Tensor4dPtr<ARM> >& ins, \
                std::vector<Tensor4dPtr<ARM> >& outs) {
    SABER_CHECK(_funcs_dense_dense.init(ins, outs, _param_dense_dense, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
template<>
Status DenseDenseHelper<ARM, Precision::INT8>::Init(OpContext<ARM> &ctx,\
        const std::vector<Tensor4dPtr<ARM> >& ins, \
                std::vector<Tensor4dPtr<ARM> >& outs) {
    SABER_CHECK(_funcs_dense_dense.init(ins, outs, _param_dense_dense, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(DenseDense, DenseDenseHelper, ARM, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(DenseDense, DenseDenseHelper, ARM, Precision::INT8);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_DENSEDENSE(X86, Precision::FP32);
template class DenseDenseHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(DenseDense, DenseDenseHelper, X86, Precision::FP32);
#ifndef USE_SGX
INSTANCE_DENSEDENSE(X86, Precision::INT8);
template class DenseDenseHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(DenseDense, DenseDenseHelper, X86, Precision::INT8);
#endif
#endif

#ifdef AMD_GPU
INSTANCE_DENSEDENSE(AMD, Precision::FP32);
template<>
Status DenseDenseHelper<AMD, Precision::FP32>::Init(OpContext<AMD> &ctx,\
        const std::vector<Tensor4dPtr<AMD> >& ins, \
                std::vector<Tensor4dPtr<AMD> >& outs) {
    SABER_CHECK(_funcs_dense_dense.init(ins, outs, _param_dense_dense, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(DenseDense, DenseDenseHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(DenseDense)
.Doc("DenseDense operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("fullconnect")
.__alias__<NV, Precision::FP32>("fc")
.__alias__<NV, Precision::INT8>("fc")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("fullconnect")
.__alias__<ARM, Precision::FP32>("fc")
.__alias__<ARM, Precision::INT8>("fc")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("fullconnect")
.__alias__<X86, Precision::FP32>("fc")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("fullconnect")
.__alias__<AMD, Precision::FP32>("fc")
#endif

.num_in(1)
.num_out(1)
.Args<int>("axis", " axis to compute ")
.Args<int>("out_dim", " out dim ")
.Args<bool>("bias_term", " whether fc weights have bias");

} /* namespace ops */

} /* namespace anakin */


