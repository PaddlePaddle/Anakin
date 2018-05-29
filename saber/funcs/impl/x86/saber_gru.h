

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
#include "saber/funcs/impl/impl_define.h"

namespace anakin {

namespace saber {

template<DataType OpDtype,
         DataType inDtype,
         DataType outDtype,
         typename LayOutType_op,
         typename LayOutType_in,
         typename LayOutType_out>
class SaberGru<X86, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase <
    Tensor<X86, inDtype, LayOutType_in>, \
    Tensor<X86, outDtype, LayOutType_out>, \
    Tensor<X86, OpDtype, LayOutType_op>, \
    GruParam<Tensor<X86, OpDtype, LayOutType_op> >> {

public:
    typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberGru() {}

    ~SaberGru() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs, \
                             std::vector<DataTensor_out*>& outputs, \
                             GruParam<OpTensor>& gru_param, Context<X86>& ctx) {
        if (_init_weights == false && gru_param._formula == GRU_ORIGIN) {
            _hidden_size = gru_param.bias()->valid_size() / 3;
            int weights_bias_size = _hidden_size * 3;
            int weights_h2h_size = _hidden_size * _hidden_size * 3;
            int weights_i2h_size = gru_param.weight()->valid_size() - weights_h2h_size;

            _word_size = weights_i2h_size / _hidden_size / 3;
            _weights_i2h.try_expand_size(weights_i2h_size);
            _weights_h2h.try_expand_size(weights_h2h_size);
            _weights_bias.try_expand_size(weights_bias_size);

            memcpy(_weights_i2h.mutable_data(), gru_param.weight()->data(),
                   sizeof(InDataType) * weights_i2h_size);
            memcpy(_weights_h2h.mutable_data(), gru_param.weight()->data() + weights_i2h_size,
                   sizeof(InDataType) * weights_h2h_size);
            memcpy(_weights_bias.mutable_data(), gru_param.bias()->data(),
                   sizeof(InDataType) * weights_bias_size);

            _temp_wh.try_expand_size(2 * _hidden_size);
            _temp_whr.try_expand_size(_hidden_size);
        }

        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs, \
                               std::vector<DataTensor_out*>& outputs, \
                               GruParam<OpTensor>& gru_param, Context<X86>& ctx) {


        return SaberSuccess;
    }


    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 GruParam<OpTensor>& param);

private:
    int _word_size;
    int _hidden_size;
    bool _init_weights = false;
    OpTensor _weights_i2h;
    OpTensor _weights_h2h;
    OpTensor _weights_bias;
    DataTensor_out _init_hidden;

    DataTensor_out _temp_wx;
    DataTensor_out _temp_wh;
    DataTensor_out _temp_whr;
//    lod_no_batch_gru(const OpDataType* weight_w, const OpDataType* weight_h,const OpDataType* b, const OutDataType* h_init, OutDataType* h_out,
//                     const InDataType* x,OutDataType *temp_wx,OutDataType *temp_wh,OutDataType *temp_whr,
//                     int hidden_size, int word_size, std::vector<int>& offset_vec, bool is_reverse);

};

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
