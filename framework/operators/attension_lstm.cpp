#include "framework/operators/attension_lstm.h"

namespace anakin {

namespace ops {

#define INSTANCE_SEQUENCE_EXPAND(Ttype, Dtype, Ptype) \
template<> \
void AttensionLstm<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<AttensionLstmHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<AttensionLstmHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_attension_lstm; \
    impl->_funcs_attension_lstm(ins, outs, param, ctx); \
}

/// TODO ... specialization other type of operator

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
AttensionLstmHelper<Ttype, Dtype, Ptype>::~AttensionLstmHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status AttensionLstmHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing AttensionLstm op parameter.";
    using pblock_type = PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>;

    auto attention_fc_w_0 = GET_PARAMETER(pblock_type, weight_1);
    auto attention_fc_b_0 = GET_PARAMETER(pblock_type, weight_2);
    auto attention_fc_w_1 = GET_PARAMETER(pblock_type, weight_3);
    auto attention_fc_b_1 = GET_PARAMETER(pblock_type, weight_4);
    auto lstm_w = GET_PARAMETER(pblock_type, weight_5);
    auto lstm_b = GET_PARAMETER(pblock_type, weight_6);

    LstmParam<Tensor4d<Ttype, Dtype>> lstm_param(&(lstm_w.d_tensor()), 
            &(lstm_b.d_tensor()), nullptr, 
            Active_unknow, Active_sigmoid, Active_tanh, Active_tanh,
            false, false, false, 1.f, 1, 1);
    FcParam<Tensor4d<Ttype, Dtype>> fc_param_0(&(attention_fc_w_0.d_tensor()),
           &(attention_fc_b_0.d_tensor()),
           attention_fc_b_0.d_tensor().valid_size());
    FcParam<Tensor4d<Ttype, Dtype>> fc_param_1(&(attention_fc_w_1.d_tensor()),
           &(attention_fc_b_1.d_tensor()),
           attention_fc_b_1.d_tensor().valid_size());
    std::vector<FcParam<Tensor4d<Ttype, Dtype>> > fc_vec;
    fc_vec.resize(2);
    fc_vec[0] = fc_param_0;
    fc_vec[1] = fc_param_1;
    AttensionParam<Tensor4d<Ttype, Dtype>> attn_param(fc_vec);
    AttensionLstmParam<Tensor4d<Ttype, Dtype>> attn_lstm_param(attn_param, lstm_param);
     
    _param_attension_lstm = attn_lstm_param;

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status AttensionLstmHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_attension_lstm.init(ins, outs, _param_attension_lstm, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status AttensionLstmHelper<Ttype, Dtype, Ptype>::InferShape(const
                                                         std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                                                         std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_attension_lstm.compute_output_shape(ins, outs, _param_attension_lstm));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SEQUENCE_EXPAND(NV, AK_FLOAT, Precision::FP32);
template<>
Status AttensionLstmHelper<NV, AK_FLOAT, Precision::FP32>::Init(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
        std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_attension_lstm.init(ins, outs, _param_attension_lstm, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(AttensionLstm, AttensionLstmHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_SEQUENCE_EXPAND(X86, AK_FLOAT, Precision::FP32);
INSTANCE_SEQUENCE_EXPAND(X86, AK_FLOAT, Precision::FP16);
INSTANCE_SEQUENCE_EXPAND(X86, AK_FLOAT, Precision::INT8);
template class AttensionLstmHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AttensionLstm, AttensionLstmHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SEQUENCE_EXPAND(ARM, AK_FLOAT, Precision::FP32);
template class AttensionLstmHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AttensionLstm, AttensionLstmHelper, ARM, AK_FLOAT, Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(AttensionLstm)
.Doc("AttensionLstm operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("attension_lstm")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("attension_lstm")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("attension_lstm")
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

