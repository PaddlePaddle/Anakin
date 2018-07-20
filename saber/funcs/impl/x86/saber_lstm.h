#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
#include "saber/funcs/impl/impl_lstm.h"

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

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             LstmParam<X86>& param,
                             Context<X86>& ctx) ;

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               LstmParam<X86>& param,
                               Context<X86>& ctx) ;

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

    template <typename BIT>
    SaberStatus avx_dispatch_without_peephole(const std::vector<Tensor<X86>*>& inputs,
                                              std::vector<Tensor<X86>*>& outputs,
                                              LstmParam<X86>& param);

    template <typename BIT>
    SaberStatus avx_dispatch_with_peephole(const std::vector<Tensor<X86>*>& inputs,
                                           std::vector<Tensor<X86>*>& outputs,
                                           LstmParam<X86>& param);

};

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
