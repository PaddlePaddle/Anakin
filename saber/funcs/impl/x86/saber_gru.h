

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
#include "saber/funcs/impl/impl_gru.h"
#include "saber/funcs/impl/x86/x86_utils.h"


#ifdef __AVX512F__
#define SABER_X86_TYPE __m512
#elif __AVX2__
#define SABER_X86_TYPE __m256
#else
#define SABER_X86_TYPE float
#endif

namespace anakin {

namespace saber {

template<DataType OpDtype>
class SaberGru<X86, OpDtype> : \
    public ImplBase <
        X86, OpDtype,GruParam<X86>> {

public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    typedef Tensor<X86> OpTensor;
    SaberGru() {}

    ~SaberGru() {}

    virtual SaberStatus init(const std::vector<OpTensor*>& inputs, \
                             std::vector<OpTensor*>& outputs, \
                             GruParam<X86>& gru_param, Context<X86>& ctx) {
        this->_ctx = &ctx;
        CHECK_EQ(gru_param.formula, GRU_ORIGIN) << "only support gru_origin now";
        _hidden_size = gru_param.bias()->valid_size() / 3;

        if (gru_param.formula == GRU_ORIGIN && _aligned_way) {
            //FIXME:aligned should be determine by framework
            int aligned_byte = sizeof(SABER_X86_TYPE);
            int c_size = aligned_byte / sizeof(OpDataType);

            _hidden_size = gru_param.bias()->valid_size() / 3;
            int weights_bias_size = _hidden_size * 3;
            int weights_h2h_size = _hidden_size * _hidden_size * 3;
            int weights_i2h_size = gru_param.weight()->valid_size() - weights_h2h_size;
            _word_size = weights_i2h_size / _hidden_size / 3;

            _aligned_size = c_size;
            _aligned_word_size = utils::round_up(_word_size, c_size);
            _aligned_hidden_size = utils::round_up(_hidden_size, c_size);
            _aligned_word_size_iter_num = _aligned_word_size / c_size;
            _aligned_hidden_size_iter_num = _aligned_hidden_size / c_size;

            Shape weights_i2h_shape({1, _word_size, 3, _aligned_hidden_size},Layout_NCHW);
            Shape weights_h2h_shape({1, _aligned_hidden_size, 3, _aligned_hidden_size},Layout_NCHW);
            Shape weights_bias_shape({1, 1, 3, _aligned_hidden_size},Layout_NCHW);
            utils::try_expand_tensor(_aligned_weights_i2h,weights_i2h_shape);
            utils::try_expand_tensor(_aligned_weights_h2h,weights_h2h_shape);
            utils::try_expand_tensor(_aligned_weights_bias,weights_bias_shape);

            utils::AlignedUtils aligned_tool;
            aligned_tool.aligned_last_dim((const OpDataType*)gru_param.weight()->data(), ( OpDataType*)_aligned_weights_i2h.mutable_data(),
                                          weights_i2h_size, _hidden_size, _aligned_hidden_size);

            aligned_tool.aligned_last_dim((const OpDataType*)gru_param.weight()->data() + weights_i2h_size,
                                          (OpDataType*) _aligned_weights_h2h.mutable_data(),
                                          weights_h2h_size, _hidden_size, _aligned_hidden_size);

            aligned_tool.aligned_last_dim((const OpDataType*)gru_param.bias()->data(), (OpDataType*)_aligned_weights_bias.mutable_data(),
                                          weights_bias_size, _hidden_size, _aligned_hidden_size);

            utils::try_expand_tensor(_weights_i2h,weights_i2h_size);
            utils::try_expand_tensor(_weights_h2h,weights_h2h_size);
            utils::try_expand_tensor(_weights_bias,weights_bias_size);

            //FIXME:format pitch
            memcpy(_weights_i2h.mutable_data(), (const OpDataType*)gru_param.weight()->data(),
                   sizeof(OpDataType) * weights_i2h_size);
            memcpy(_weights_h2h.mutable_data(), (const OpDataType*)gru_param.weight()->data() + weights_i2h_size,
                   sizeof(OpDataType) * weights_h2h_size);
            memcpy(_weights_bias.mutable_data(), (const OpDataType*)gru_param.bias()->data(),
                   sizeof(OpDataType) * weights_bias_size);

        } else if (gru_param.formula == GRU_ORIGIN) {
            _hidden_size = gru_param.bias()->valid_size() / 3;
            int weights_bias_size = _hidden_size * 3;
            int weights_h2h_size = _hidden_size * _hidden_size * 3;
            int weights_i2h_size = gru_param.weight()->valid_size() - weights_h2h_size;
            _word_size = weights_i2h_size / _hidden_size / 3;

            utils::try_expand_tensor(_weights_i2h,weights_i2h_size);
            utils::try_expand_tensor(_weights_h2h,weights_h2h_size);
            utils::try_expand_tensor(_weights_bias,weights_bias_size);

            memcpy(_weights_i2h.mutable_data(), (const OpDataType*)gru_param.weight()->data(),
                   sizeof(OpDataType) * weights_i2h_size);
            memcpy(_weights_h2h.mutable_data(), (const OpDataType*)gru_param.weight()->data() + weights_i2h_size,
                   sizeof(OpDataType) * weights_h2h_size);
            memcpy(_weights_bias.mutable_data(), (const OpDataType*)gru_param.bias()->data(),
                   sizeof(OpDataType) * weights_bias_size);
        }

        LOG(INFO) << "success init";
        return create(inputs, outputs, gru_param, ctx);
    }

    virtual SaberStatus create(const std::vector<OpTensor*>& inputs, \
                               std::vector<OpTensor*>& outputs, \
                               GruParam<X86>& gru_param, Context<X86>& ctx) {


        return SaberSuccess;
    }


    virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                                 std::vector<OpTensor*>& outputs,
                                 GruParam<X86>& param);

private:
    int _word_size;
    int _hidden_size;

    bool _aligned_way = true;
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
                                GruParam<X86>& param);

};

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_GRU_H
