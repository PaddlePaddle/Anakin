#include "framework/operators/lstm.h"
#include <unordered_map>
namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Lstm<NV, Precision::FP32>::operator() (OpContext<NV> &ctx,
                          const std::vector<Tensor4dPtr<NV> >& ins,
                          std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<LstmHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<LstmHelper<NV, Precision::FP32>*>(this->_helper)->_param_lstm;
    impl->_funcs_lstm(ins, outs, param, ctx);
}
#endif
#ifdef USE_X86_PLACE
template<>
void Lstm<X86, Precision::FP32>::operator() (OpContext<X86> &ctx,
                                                     const std::vector<Tensor4dPtr<X86> >& ins,
                                                     std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl = static_cast<LstmHelper<X86, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<LstmHelper<X86, Precision::FP32>*>(this->_helper)->_param_lstm;
    impl->_funcs_lstm(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator
/// set helper
template<typename Ttype, Precision Ptype>
LstmHelper<Ttype, Ptype>::~LstmHelper() {
}

template<typename Ttype, Precision Ptype>
Status LstmHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Lstm op parameter.";

    auto num_direction = GET_PARAMETER(int, num_direction);
    auto dropout_param = GET_PARAMETER(float, dropout_param);
    auto num_layers = GET_PARAMETER(int, num_layers);
    auto input_activation = GET_PARAMETER(std::string, input_activation);
    auto gate_activation = GET_PARAMETER(std::string, gate_activation);
    auto cell_activation = GET_PARAMETER(std::string, cell_activation);
    auto candidate_activation = GET_PARAMETER(std::string, candidate_activation);
    auto is_reverse = GET_PARAMETER(bool, is_reverse);
    auto use_peepholes = GET_PARAMETER(bool, use_peepholes);

    //auto weight_wu = GET_PARAMETER(PBlock<typename DataTypeWarpper<pe>::type>, weight_1);
    //auto bias = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_2);
    using pblock_type = PBlock<Ttype>;
    auto weight_wu = GET_PARAMETER(pblock_type, weight_1);
    auto bias = GET_PARAMETER(pblock_type, weight_2);


    LOG(INFO)<<"lstm act = ["<<input_activation<<","<<gate_activation<<","<<cell_activation<<","<<candidate_activation<<"]";
    LOG(INFO)<<"lstm other param = ["<<use_peepholes<<","<<is_reverse<<","<<dropout_param<<","<<num_direction<<","<<num_layers<<"]";

    std::unordered_map<std::string, ActiveType> enum_map = {
            {"null",Active_unknow},
            {"sigmoid_fluid", Active_sigmoid},
            {"relu_fluid", Active_relu},
            {"tanh_fluid", Active_tanh},
            {"identity_fluid", Active_identity},
            {"sigmoid", Active_sigmoid},
            {"tanh", Active_tanh},
    };
    LstmParam<Ttype> lstm_param(&(weight_wu.d_tensor()), &(bias.d_tensor()), nullptr,
            enum_map[input_activation], enum_map[gate_activation],
            enum_map[cell_activation], enum_map[candidate_activation],
            use_peepholes, false, is_reverse, dropout_param,
            num_direction, num_layers);
    _param_lstm = lstm_param;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status LstmHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx,
                                                const std::vector<Tensor4dPtr<Ttype> >& ins,
                                                std::vector<Tensor4dPtr<Ttype> >& outs) {
    DLOG(INFO)<<"inti lstm in op.cpp";
    SABER_CHECK(_funcs_lstm.init(ins, outs, _param_lstm, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status LstmHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                                                      std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_lstm.compute_output_shape(ins, outs, _param_lstm));
    return Status::OK();
}

#ifdef USE_CUDA
template class LstmHelper<NV, Precision::FP32>;
template class LstmHelper<NV, Precision::FP16>;
template class LstmHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class LstmHelper<ARM, Precision::FP32>;
template class LstmHelper<ARM, Precision::FP16>;
template class LstmHelper<ARM, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
template class LstmHelper<X86, Precision::FP32>;
template class LstmHelper<X86, Precision::FP16>;
template class LstmHelper<X86, Precision::INT8>;
#endif

#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Lstm, LstmHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Lstm, LstmHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(Lstm, LstmHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Lstm)
    .Doc("Lstm operator")
#ifdef USE_CUDA
    .__alias__<NV, Precision::FP32>("Lstm")
    .__alias__<NV, Precision::FP32>("LSTM")
#endif
#ifdef USE_ARM_PLACE
    .__alias__<ARM, Precision::FP32>("Lstm")
#endif
#ifdef USE_X86_PLACE
    .__alias__<X86, Precision::FP32>("Lstm")
    .__alias__<X86, Precision::FP32>("LSTM")
#endif
    .num_in(1)
    .num_out(1)
    .Args<bool>("is_reverse", " is_reverse for lstm.")
    .Args<int>("num_direction", "some descp")
    .Args<float>("dropout_param", "some descp")
    .Args<int>("num_layers", "some descp")
    .Args<std::string>("input_activation", "some descp")
    .Args<std::string>("gate_activation", "some descp")
    .Args<std::string>("cell_activation", "some descp")
    .Args<std::string>("candidate_activation", "some descp")
    .Args<bool>("is_reverse", "some descp")
    .Args<bool>("use_peephole", "some descp");

} /* namespace ops */

} /* namespace anakin */


