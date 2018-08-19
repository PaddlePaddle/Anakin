

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
#include "saber/funcs/impl/impl_gru.h"
#include "saber/funcs/impl/x86/x86_utils.h"


#if defined(__AVX512F__)
#define SABER_X86_TYPE __m512
#elif defined(__AVX2__)
#define SABER_X86_TYPE __m256
#elif defined(__SSE4_2__)
#define SABER_X86_TYPE __m128
#else
#define SABER_X86_TYPE float
#endif

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
        this->_ctx = &ctx;
        CHECK_EQ(gru_param.formula ,GRU_ORIGIN)<<"only support gru_origin now";
        _hidden_size = gru_param.bias()->valid_size() / 3;
        if (gru_param.formula == GRU_ORIGIN&&_aligned_way) {
            //FIXME:aligned should be determine by framework
            int aligned_byte= sizeof(SABER_X86_TYPE);
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
        }else if(gru_param.formula == GRU_ORIGIN){
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

//    typedef  __m256 _aligned_type;
    bool _aligned_way=true;
    int _aligned_word_size;
    int _aligned_hidden_size;
    int _aligned_size;
    int _aligned_word_size_iter_num;
    int _aligned_hidden_size_iter_num;

    OpTensor _weights_i2h;
    OpTensor _weights_h2h;
    OpTensor _weights_bias;
    OpTensor _init_hidden;

    OpTensor _aligned_weights_i2h;
    OpTensor _aligned_weights_h2h;
    OpTensor _aligned_weights_bias;
    OpTensor _aligned_init_hidden;

    OpTensor _temp_wx;
    OpTensor _temp_wh;
    OpTensor _temp_whr;

    OpTensor _temp_x;
    OpTensor _temp_out;
    OpTensor _temp_h_init;

    template <typename BIT>
    SaberStatus batch_s_aligned(\
                                const std::vector<OpTensor*>& inputs,
                                std::vector<OpTensor*>& outputs,
                                GruParam<OpTensor>& param);

};

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
