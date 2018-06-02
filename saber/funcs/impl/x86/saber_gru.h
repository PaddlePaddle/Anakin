

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
#include "saber/funcs/impl/impl_gru.h"
#include "saber/funcs/impl/x86/x86_utils.h"
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
        this->_ctx=ctx;
        CHECK_EQ(gru_param._formula ,GRU_ORIGIN)<<"only support gru_origin now";
        _hidden_size = gru_param.bias()->valid_size() / 3;
        if (gru_param._formula == GRU_ORIGIN&&_aligned_way) {
            //FIXME:aligned should be determine by framework
            int aligned_byte=64;
            int c_size=aligned_byte/sizeof(OpDataType);

            _hidden_size = gru_param.bias()->valid_size() / 3;
            int weights_bias_size = _hidden_size * 3;
            int weights_h2h_size = _hidden_size * _hidden_size * 3;
            int weights_i2h_size = gru_param.weight()->valid_size() - weights_h2h_size;
            _word_size = weights_i2h_size / _hidden_size / 3;

            _aligned_size=c_size;
            _aligned_word_size=utils::round_up(_word_size,c_size);
            _aligned_hidden_size=utils::round_up(_hidden_size,c_size);
            _aligned_word_size_iter_num=_aligned_word_size/c_size;
            _aligned_hidden_size_iter_num=_aligned_hidden_size/c_size;

            Shape weights_i2h_shape(1,_word_size,3,_aligned_hidden_size);
            Shape weights_h2h_shape(1,_aligned_hidden_size,3,_aligned_hidden_size);
            Shape weights_bias_shape(1,1,3,_aligned_hidden_size);
            _aligned_weights_i2h.try_expand_size(weights_i2h_shape);
            _aligned_weights_h2h.try_expand_size(weights_h2h_shape);
            _aligned_weights_bias.try_expand_size(weights_bias_shape);

            utils::AlignedUtils aligned_tool;
            aligned_tool.aligned_last_dim(gru_param.weight()->data(),_aligned_weights_i2h.mutable_data(),
                                          weights_i2h_size,_hidden_size,_aligned_hidden_size);

            aligned_tool.aligned_last_dim(gru_param.weight()->data() + weights_i2h_size,_aligned_weights_h2h.mutable_data(),
                                          weights_h2h_size,_hidden_size,_aligned_hidden_size);

            aligned_tool.aligned_last_dim(gru_param.bias()->data(),_aligned_weights_bias.mutable_data(),
                                          weights_bias_size,_hidden_size,_aligned_hidden_size);

            _weights_i2h.try_expand_size(weights_i2h_size);
            _weights_h2h.try_expand_size(weights_h2h_size);
            _weights_bias.try_expand_size(weights_bias_size);
            //FIXME:format pitch
            memcpy(_weights_i2h.mutable_data(), gru_param.weight()->data(),
                   sizeof(InDataType) * weights_i2h_size);
            memcpy(_weights_h2h.mutable_data(), gru_param.weight()->data() + weights_i2h_size,
                   sizeof(InDataType) * weights_h2h_size);
            memcpy(_weights_bias.mutable_data(), gru_param.bias()->data(),
                   sizeof(InDataType) * weights_bias_size);

//            Shape wh_shape(1,1,2,_aligned_hidden_size/c_size,c_size);
//            Shape whr_shape(1,1,1,_aligned_hidden_size/c_size,c_size);
//            _temp_wh.try_expand_size(wh_shape);
//            _temp_whr.try_expand_size(whr_shape);
        }else if(gru_param._formula == GRU_ORIGIN){
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

        }
        LOG(INFO)<<"success init";
        return create(inputs,outputs,gru_param,ctx);
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

    bool _aligned_way=true;
    int _aligned_word_size;
    int _aligned_hidden_size;
    int _aligned_size;
    int _aligned_word_size_iter_num;
    int _aligned_hidden_size_iter_num;

    OpTensor _weights_i2h;
    OpTensor _weights_h2h;
    OpTensor _weights_bias;
    DataTensor_out _init_hidden;

    OpTensor _aligned_weights_i2h;
    OpTensor _aligned_weights_h2h;
    OpTensor _aligned_weights_bias;
    DataTensor_out _aligned_init_hidden;

    DataTensor_out _temp_wx;
    DataTensor_out _temp_wh;
    DataTensor_out _temp_whr;

    DataTensor_in _temp_x;
    DataTensor_out _temp_out;
    DataTensor_out _temp_h_init;
//    lod_no_batch_gru(const OpDataType* weight_w, const OpDataType* weight_h,const OpDataType* b, const OutDataType* h_init, OutDataType* h_out,
//                     const InDataType* x,OutDataType *temp_wx,OutDataType *temp_wh,OutDataType *temp_whr,
//                     int hidden_size, int word_size, std::vector<int>& offset_vec, bool is_reverse);
    SaberStatus naiv_gru(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        GruParam<OpTensor>& param);

    SaberStatus batch_gru(\
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        GruParam<OpTensor>& param);

    SaberStatus naiv_256(\
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        GruParam<OpTensor>& param);

    SaberStatus naiv_256_s_aligned(\
    const std::vector<DataTensor_in*>& inputs,
               std::vector<DataTensor_out*>& outputs,
               GruParam<OpTensor>& param);

    SaberStatus batch_256_s_aligned(\
    const std::vector<DataTensor_in*>& inputs,
                       std::vector<DataTensor_out*>& outputs,
                       GruParam<OpTensor>& param);
};

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
