#include "framework/operators/lstmp.h"
#include <unordered_map>
namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Lstmp<NV, Precision::FP32>::operator()(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<LstmpHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<LstmpHelper<NV, Precision::FP32>*>(this->_helper)->_param_lstm;
    impl->_funcs_lstm(ins, outs, param, ctx);
}
#endif
#ifdef USE_X86_PLACE
template<>
void Lstmp<X86, Precision::FP32>::operator()(OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl = static_cast<LstmpHelper<X86, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<LstmpHelper<X86, Precision::FP32>*>(this->_helper)->_param_lstm;
    impl->_funcs_lstm(ins, outs, param, ctx);
}
template<>
void Lstmp<X86, Precision::INT8>::operator()(OpContext<X86>& ctx,
                                             const std::vector<Tensor4dPtr<X86> >& ins,
                                             std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl = static_cast<LstmpHelper<X86, Precision::INT8>*>(this->_helper);
    auto& param = static_cast<LstmpHelper<X86, Precision::INT8>*>(this->_helper)->_param_lstm;
    impl->_funcs_lstm(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator
/// set helper
template<typename Ttype, Precision Ptype>
LstmpHelper<Ttype, Ptype>::~LstmpHelper() {
}

template<typename Ttype, Precision Ptype>
Status LstmpHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Lstm op parameter.";

    auto cell_dim = GET_PARAMETER(int, cellDim);
    auto skip_num = GET_PARAMETER(int, skipNum);
    auto out_dim = GET_PARAMETER(int, outDim);
    auto rec_act_type = GET_PARAMETER(std::string, recActType);


    using pblock_type = PBlock<Ttype>;
    auto weight_wu = GET_PARAMETER(pblock_type, weight_1);
    auto bias = GET_PARAMETER(pblock_type, weight_2);


    LOG(INFO) << "lstmp args = [" << cell_dim << "," << out_dim << "," << skip_num
              << "," << rec_act_type << "]";

    const bool use_peepholes= true;
    bool with_peephole_in = true;
    bool skip_input_in = false;
    bool is_reverse_in = false;
    float dropout_param_in = 1.f;
    int num_direction_in = 1;
    int numLayers_in = 1;
    LstmParam<Ttype> lstm_param(&(weight_wu.d_tensor()), &(bias.d_tensor()), nullptr,
                                Active_unknow, Active_sigmoid,
                                Active_tanh, Active_tanh,
                                with_peephole_in, skip_input_in, is_reverse_in, dropout_param_in,
                                num_direction_in, numLayers_in,skip_num,out_dim,cell_dim);
    _param_lstm = lstm_param;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status LstmpHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
                                      const std::vector<Tensor4dPtr<Ttype> >& ins,
                                      std::vector<Tensor4dPtr<Ttype> >& outs) {
    DLOG(INFO) << "inti lstm in op.cpp";
    SABER_CHECK(_funcs_lstm.init(ins, outs, _param_lstm, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status LstmpHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_lstm.compute_output_shape(ins, outs, _param_lstm));
    return Status::OK();
}

#ifdef USE_CUDA
template class LstmpHelper<NV, Precision::FP32>;
template class LstmpHelper<NV, Precision::FP16>;
template class LstmpHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class LstmpHelper<ARM, Precision::FP32>;
template class LstmpHelper<ARM, Precision::FP16>;
template class LstmpHelper<ARM, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
template class LstmpHelper<X86, Precision::FP32>;
template class LstmpHelper<X86, Precision::FP16>;
template class LstmpHelper<X86, Precision::INT8>;
#endif

#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Lstmp, LstmpHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Lstmp, LstmpHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(Lstmp, LstmpHelper, X86, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Lstmp, LstmpHelper, X86, Precision::INT8);
#endif
//! register op
ANAKIN_REGISTER_OP(Lstmp)
.Doc("Lstmp operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("Lstmp")
.__alias__<NV, Precision::FP32>("LSTMP")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("Lstmp")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("Lstmp")
.__alias__<X86, Precision::FP32>("LSTMP")
.__alias__<X86, Precision::INT8>("Lstmp")
.__alias__<X86, Precision::INT8>("LSTMP")
#endif
.num_in(1)
.num_out(1)
.Args<int>("cellDim", " is_reverse for lstm.")
.Args<int>("skipNum", "some descp")
.Args<int>("outDim", "some descp")
.Args<bool>("recActType", "some descp");

} /* namespace ops */

} /* namespace anakin */


