#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
#include "saber/funcs/impl/impl_lstm.h"
#include "saber_funcs_param.h"
#include "saber/funcs/impl/x86/x86_utils.h"


#if defined(__AVX512F__)
#define SABER_X86_TYPE __m512
#elif defined(__AVX2__) and defined(__FMA__)
#define SABER_X86_TYPE __m256
#elif defined(__SSE4_2__) and defined(__FMA__)
#define SABER_X86_TYPE __m128
#else
#define SABER_X86_TYPE float
#endif

//#define SABER_X86_TYPE __m128

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberLstm<X86, OpDtype> :
    public ImplBase <
    X86, OpDtype,LstmParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
//    typedef Tensor<X86> OpTensor;
    SaberLstm() {}

    ~SaberLstm() {}

    SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             LstmParam<X86>& param,
                             Context<X86>& ctx){
        if(param.with_peephole){
        _hidden_size=param.bias()->valid_size()/7;
        }else{
            _hidden_size=param.bias()->valid_size()/4;
        }
        _word_size=(param.weight()->valid_size()-_hidden_size*_hidden_size*4)/_hidden_size/4;

        int weights_i2h_size=4*_hidden_size*_word_size;
        int weights_h2h_size=4*_hidden_size*_hidden_size;
        int weights_bias_size=4*_hidden_size;
        int weights_peephole_size=3*_hidden_size;

        int aligned_byte= sizeof(SABER_X86_TYPE);
        int c_size=aligned_byte/sizeof(OpDataType);

        _aligned_word_size=utils::round_up(_word_size,c_size);
        _aligned_hidden_size=utils::round_up(_hidden_size,c_size);


        Shape aligned_weights_i2h_shape({1,_word_size,4,_aligned_hidden_size});
        Shape aligned_weights_h2h_shape({1,_aligned_hidden_size,4,_aligned_hidden_size});
        Shape aligned_weights_bias_shape({1,1,4,_aligned_hidden_size});
        utils::try_expand_tensor(_aligned_weights_i2h,aligned_weights_i2h_shape);
        utils::try_expand_tensor(_aligned_weights_h2h,aligned_weights_h2h_shape);
        utils::try_expand_tensor(_aligned_weights_bias,aligned_weights_bias_shape);

        utils::AlignedUtils aligned_tool;
        aligned_tool.aligned_last_dim((OpDataType*)(param.weight()->data()),(OpDataType*)_aligned_weights_i2h.mutable_data(),
                weights_i2h_size,_hidden_size,_aligned_hidden_size);

        aligned_tool.aligned_last_dim((OpDataType*)(param.weight()->data()) + weights_i2h_size,(OpDataType*)_aligned_weights_h2h.mutable_data(),
                weights_h2h_size,_hidden_size,_aligned_hidden_size);

        aligned_tool.aligned_last_dim((OpDataType*)param.bias()->data(),(OpDataType*)_aligned_weights_bias.mutable_data(),
                weights_bias_size,_hidden_size,_aligned_hidden_size);
        //FIXME:init weights tensor
        if(param.with_peephole){
        Shape aligned_weights_peephole_shape({1,1,3,_aligned_hidden_size});
        utils::try_expand_tensor(_aligned_weights_peephole,aligned_weights_peephole_shape);
        aligned_tool.aligned_last_dim((OpDataType*)(param.bias()->data())+weights_bias_size,(OpDataType*)_aligned_weights_peephole.mutable_data(),
                weights_peephole_size,_hidden_size,_aligned_hidden_size);
        }

        return create(inputs,outputs,param,ctx);
    } ;

    SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               LstmParam<X86>& param,
                               Context<X86>& ctx) {
        return SaberSuccess;
    };

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 LstmParam<X86>& param) ;

private:

    int _word_size;
    int _hidden_size;
    int _aligned_word_size;
    int _aligned_hidden_size;


    Tensor<X86> _weights_i2h;
    Tensor<X86> _weights_h2h;
    Tensor<X86> _weights_bias;
    Tensor<X86> _weights_peephole;
    Tensor<X86> _init_hidden;

    Tensor<X86> _aligned_weights_i2h;
    Tensor<X86> _aligned_weights_h2h;
    Tensor<X86> _aligned_weights_bias;
    Tensor<X86> _aligned_weights_peephole;

    Tensor<X86> _aligned_init_hidden;

    Tensor<X86> _temp_wx;
    Tensor<X86> _temp_wh;
    Tensor<X86> _temp_cell;

    Tensor<X86> _temp_x;
    Tensor<X86> _temp_out;
    Tensor<X86> _temp_h_init;

    template <typename BIT,bool with_peephole >
    SaberStatus avx_dispatch(const std::vector<Tensor<X86>*>& inputs,
                                              std::vector<Tensor<X86>*>& outputs,
                                              LstmParam<X86>& param);


};

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
