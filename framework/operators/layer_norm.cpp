#include "framework/operators/layer_norm.h"

namespace anakin{

namespace ops{

#define INSTANCE_LAYER_NORM(Ttype, Ptype) \
template<> \
void LayerNorm<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<LayerNormHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<LayerNormHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_layer_norm; \
    impl->_funcs_layer_norm(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status LayerNormHelper<Ttype, Ptype>::InitParam() {
    auto axis = GET_PARAMETER(int, begin_norm_axis);
    auto eps = GET_PARAMETER(float, eps);

	using pblock_type = PBlock<Ttype>;
    auto input_scale = GET_PARAMETER(pblock_type, weight_1);
    auto input_bias = GET_PARAMETER(pblock_type, weight_2);

    saber::LayerNormParam<Ttype> param(axis, eps, &(input_scale.d_tensor()), \
            &(input_bias.d_tensor()));
    _param_layer_norm = param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status LayerNormHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx,
const std::vector<Tensor4dPtr<Ttype> >& ins,
std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_layer_norm.init(ins, outs, _param_layer_norm, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status LayerNormHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_layer_norm.compute_output_shape(ins, outs, _param_layer_norm));
    return Status::OK();
}


#ifdef USE_CUDA
INSTANCE_LAYER_NORM(NV, Precision::FP32);
template class LayerNormHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(LayerNorm, LayerNormHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_LAYER_NORM(AMD, Precision::FP32);
template class LayerNormHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(LayerNorm, LayerNormHelper, AMD, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_LAYER_NORM(X86, Precision::FP32);
template class LayerNormHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(LayerNorm, LayerNormHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_LAYER_NORM(ARM, Precision::FP32);
template class LayerNormHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(LayerNorm, LayerNormHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(LayerNorm)
.Doc("LayerNorm operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("layernorm")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("layernorm")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("layernorm")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("layernorm")
#endif
.num_in(1)
.num_out(1)
.Args<int>("begin_norm_axis", " begin norm axis")
.Args<float>("eps", "eps");

} //ops

} //anakin
