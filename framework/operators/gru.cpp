#include "framework/operators/gru.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Gru<NV, AK_FLOAT, Precision::FP32>::operator() (OpContext<NV> &ctx, 
                          const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins, 
                          std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl = static_cast<GruHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<GruHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_gru;
    impl->_funcs_gru(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator
#ifdef USE_ARM_PLACE
template<>
void Gru<ARM, AK_FLOAT, Precision::FP32>::operator() (OpContext<ARM> &ctx,
                          const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins,
                          std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    auto* impl = static_cast<GruHelper<ARM, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<GruHelper<ARM, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_gru;
    impl->_funcs_gru(ins, outs, param, ctx);
}
#endif

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
GruHelper<Ttype, Dtype, Ptype>::~GruHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status GruHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Gru op parameter.";
    auto is_reverse = GET_PARAMETER(bool, is_reverse);
    auto gate_act = GET_PARAMETER(std::string, gate_activation);
    auto hidden_act = GET_PARAMETER(std::string, activation);
    auto formula = GET_PARAMETER(std::string, gru_formula);

//    auto weight_h2h = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_1);
//    auto bias = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_3);
//    auto weight_i2h = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_2);

    auto weight_wu = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_1);
    auto bias = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_2);

    CHECK((formula != "") && (formula == "gru_origin"
                              || formula == "gru_cudnn")) << "formula illegal";

    std::unordered_map<std::string, ActiveType> act_map = {
            {"sigmoid_fluid", Active_sigmoid_fluid},
            {"relu_fluid", Active_relu},
            {"tanh_fluid", Active_tanh_fluid},
            {"identity_fluid", Active_identity}
    };
    std::unordered_map<std::string, GruFormula > formula_map = {
            {"gru_origin", GRU_ORIGIN},
            {"gru_cudnn", GRU_CUDNN},
    };
    CHECK_GT(weight_wu.d_tensor().valid_size(),0)<<"weights size must > 0";
    CHECK_GT(bias.d_tensor().valid_size(),0)<<"bias size must > 0";

    GruParam<Tensor4d<Ttype, Dtype>> gru_param(&(weight_wu.d_tensor()), &(bias.d_tensor()),
                                               formula_map[formula], act_map[gate_act],
                                               act_map[hidden_act], is_reverse);

    _param_gru = gru_param;

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status GruHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx, 
                                                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                                                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_gru.init(ins, outs, _param_gru, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status GruHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                                                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_gru.compute_output_shape(ins, outs, _param_gru));
    return Status::OK();
}

#ifdef USE_CUDA
template class GruHelper<NV, AK_FLOAT, Precision::FP32>;
template class GruHelper<NV, AK_FLOAT, Precision::FP16>;
template class GruHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class GruHelper<ARM, AK_FLOAT, Precision::FP32>;
#endif
#ifdef ANAKIN_TYPE_FP16
template class GruHelper<ARM, AK_FLOAT, Precision::FP16>;
#endif
#ifdef ANAKIN_TYPE_INT8
template class GruHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif
#endif


#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Gru, GruHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Gru, GruHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Gru)
    .Doc("Gru operator")
#ifdef USE_CUDA
    .__alias__<NV, AK_FLOAT, Precision::FP32>("gru")
#endif
#ifdef USE_ARM_PLACE
    .__alias__<ARM, AK_FLOAT, Precision::FP32>("gru")
#endif
    .num_in(1)
    .num_out(1)
    .Args<bool>("is_reverse", " is_reverse for gru.")
    .Args<std::string>("gate_activation",  "gate_activation for gru.")
    .Args<std::string>("activation", "hidden_activation for gru.");

} /* namespace ops */

} /* namespace anakin */


