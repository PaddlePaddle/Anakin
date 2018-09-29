#include "framework/operators/normalize.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Normalize<NV, AK_FLOAT, Precision::FP32>::operator() (
	OpContext<NV> &ctx, 
	const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins, 
	std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl = static_cast<NormalizeHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<NormalizeHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_normalize;
    impl->_funcs_normalize(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
NormalizeHelper<Ttype, Dtype, Ptype>::~NormalizeHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status NormalizeHelper<Ttype, Dtype, Ptype>::InitParam() {
    //DLOG(WARNING) << "Parsing Normalize op parameter.";
    auto is_across_spatial = GET_PARAMETER(bool, is_across_spatial);
    auto is_shared_channel = GET_PARAMETER(bool, is_shared_channel);
    auto eps = GET_PARAMETER(float, eps);
    auto p = GET_PARAMETER(int, p);

    if (FIND_PARAMETER(weight_1)) {
        using pblock_type = PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>;
        auto input_scale = GET_PARAMETER(pblock_type, weight_1);
        saber::NormalizeParam<Tensor4d<Ttype, Dtype>> normalize_param(is_across_spatial, is_shared_channel, \
            &(input_scale.d_tensor()), eps, p);
        _param_normalize = normalize_param;
    } else {
        saber::NormalizeParam<Tensor4d<Ttype, Dtype>> normalize_param(is_across_spatial, is_shared_channel, eps, p);
        _param_normalize = normalize_param;
    }
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status NormalizeHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx, 
                                                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                                                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_normalize.init(ins, outs, _param_normalize, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status NormalizeHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                                                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
   SABER_CHECK(_funcs_normalize.compute_output_shape(ins, outs, _param_normalize));
   return Status::OK();
}

#ifdef USE_CUDA
template class NormalizeHelper<NV, AK_FLOAT, Precision::FP32>;
template class NormalizeHelper<NV, AK_FLOAT, Precision::FP16>;
template class NormalizeHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class NormalizeHelper<ARM, AK_FLOAT, Precision::FP32>;
template class NormalizeHelper<ARM, AK_FLOAT, Precision::FP16>;
template class NormalizeHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

// register helper 
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Normalize, NormalizeHelper, NV, AK_FLOAT, Precision::FP32);
#endif 

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Normalize, NormalizeHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Normalize)
    .Doc("Normalize operator")
#ifdef USE_CUDA
    .__alias__<NV, AK_FLOAT, Precision::FP32>("normalize")
#endif
#ifdef USE_ARM_PLACE
    .__alias__<ARM, AK_FLOAT, Precision::FP32>("normalize")
#endif
    .num_in(1)
    .num_out(1)
    .Args<bool>("is_across_spatial", "")
    .Args<bool>("is_shared_channel", "")
    .Args<float>("eps", "")
    .Args<int>("p", "");

} /* namespace ops */

} /* namespace anakin */


