#include "framework/operators/gru.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Gru<NV, Precision::FP32>::operator()(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<GruHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<GruHelper<NV, Precision::FP32>*>(this->_helper)->_param_gru;
    impl->_funcs_gru(ins, outs, param, ctx);
}
#endif
#ifdef USE_X86_PLACE
template<>
void Gru<X86, Precision::FP32>::operator()(OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl = static_cast<GruHelper<X86, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<GruHelper<X86, Precision::FP32>*>(this->_helper)->_param_gru;
    impl->_funcs_gru(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator
/// set helper
template<typename Ttype, Precision Ptype>
GruHelper<Ttype, Ptype>::~GruHelper() {
}

template<typename Ttype, Precision Ptype>
Status GruHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Gru op parameter.";
    auto is_reverse = GET_PARAMETER(bool, is_reverse);
    auto gate_act = GET_PARAMETER(std::string, gate_activation);
    auto hidden_act = GET_PARAMETER(std::string, activation);
    auto formula = GET_PARAMETER(std::string, gru_formula);

    using pblock_type = PBlock<Ttype>;
    auto weight_wu = GET_PARAMETER(pblock_type, weight_1);
    auto bias = GET_PARAMETER(pblock_type, weight_2);

    CHECK((formula != "") && (formula == "gru_origin"
                              || formula == "gru_cudnn")) << "formula illegal";

    std::unordered_map<std::string, ActiveType> act_map = {
        {"sigmoid", Active_sigmoid},
        {"sigmoid_fluid", Active_sigmoid},
        {"relu_fluid", Active_relu},
        {"tanh_fluid", Active_tanh},
        {"tanh", Active_tanh},
        {"identity_fluid", Active_identity}
    };
    std::unordered_map<std::string, GruFormula > formula_map = {
        {"gru_origin", GRU_ORIGIN},
        {"gru_cudnn", GRU_CUDNN},
    };
    CHECK_GT(weight_wu.d_tensor().valid_size(), 0) << "weights size must > 0";
    CHECK_GT(bias.d_tensor().valid_size(), 0) << "bias size must > 0";

    GruParam<Ttype> gru_param(&(weight_wu.d_tensor()), &(bias.d_tensor()),
                              formula_map[formula], act_map[gate_act],
                              act_map[hidden_act], is_reverse);

    _param_gru = gru_param;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status GruHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
                                     const std::vector<Tensor4dPtr<Ttype> >& ins,
                                     std::vector<Tensor4dPtr<Ttype> >& outs) {
    if(std::is_same<Ttype,NV>::value&&_param_gru.formula==GRU_CUDNN){
        SABER_CHECK(_funcs_gru.init(ins, outs, _param_gru, SPECIFY, VENDER_IMPL, ctx));
    }else{
        SABER_CHECK(_funcs_gru.init(ins, outs, _param_gru, SPECIFY, SABER_IMPL, ctx));
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status GruHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_gru.compute_output_shape(ins, outs, _param_gru));
    return Status::OK();
}

#ifdef USE_CUDA
template class GruHelper<NV, Precision::FP32>;
template class GruHelper<NV, Precision::FP16>;
template class GruHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Gru, GruHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
template class GruHelper<ARM, Precision::FP32>;
template class GruHelper<ARM, Precision::FP16>;
template class GruHelper<ARM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Gru, GruHelper, ARM, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
template class GruHelper<X86, Precision::FP32>;
template class GruHelper<X86, Precision::FP16>;
template class GruHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Gru, GruHelper, X86, Precision::FP32);
#endif


//! register op
ANAKIN_REGISTER_OP(Gru)
.Doc("Gru operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("gru")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("gru")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("gru")
#endif
.num_in(1)
.num_out(1)
.Args<bool>("is_reverse", " is_reverse for gru.")
.Args<std::string>("gate_activation",  "gate_activation for gru.")
.Args<std::string>("activation", "hidden_activation for gru.");

} /* namespace ops */

} /* namespace anakin */


