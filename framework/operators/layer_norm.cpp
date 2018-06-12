#include "framework/operators/layer_norm.h"

namespace anakin{

namespace ops{

#define INSTANCE_LAYERNORM(Ttype, Dtype, Ptype) \
template<> \
void LayerNorm<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<LayerNormHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<LayerNormHelper<Ttype, Dtype, Ptype>*> \
                  (this->_helper)->_param_layer_norm; \
    impl->_funcs_layer_norm(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status LayerNormHelper<Ttype, Dtype, Ptype>::InitParam() {
    auto axis = GET_PARAMETER(int, begin_norm_axis);
    auto eps = GET_PARAMETER(float, eps);

    auto input_scale = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_1);
    auto input_bias = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_2);

    saber::LayerNormParam<Tensor4d<Ttype, Dtype>> param(axis, eps, &(input_scale.d_tensor()), \
            &(input_bias.d_tensor()));
    _param_layer_norm = param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status LayerNormHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx,
const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_layer_norm.init(ins, outs, _param_layer_norm, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status LayerNormHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_layer_norm.compute_output_shape(ins, outs, _param_layer_norm));
    return Status::OK();
}


#ifdef USE_CUDA
INSTANCE_LAYERNORM(NV, AK_FLOAT, Precision::FP32);
template class LayerNormHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(LayerNorm, LayerNormHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_LAYERNORM(X86, AK_FLOAT, Precision::FP32);
template class LayerNormHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(LayerNorm, LayerNormHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_LAYERNORM(ARM, AK_FLOAT, Precision::FP32);
template class LayerNormHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(LayerNorm, LayerNormHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(LayerNorm)
.Doc("LayerNorm operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("layernorm")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("layernorm")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("layernorm")
#endif
.num_in(1)
.num_out(1)
.Args<int>("begin_norm_axis", " begin norm axis")
.Args<float>("eps", "eps");

} //ops

} //anakin