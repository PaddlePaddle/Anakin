#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTMP_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTMP_H
#include "saber/funcs/impl/impl_lstmp.h"
#include "saber_funcs_param.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/saber_lstm.h"
#include "saber/funcs/impl/x86/mkl_gemm.h"
#include "saber/funcs/impl/x86/intrinsic_packed_fc.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#define SABER_X86_TYPE __m512
#elif defined(__AVX2__) and defined(__FMA__)
#include <immintrin.h>
#define SABER_X86_TYPE __m256
#elif defined(__SSE4_2__) and defined(__FMA__)
#include <immintrin.h>
#define SABER_X86_TYPE __m128
#else
#define SABER_X86_TYPE float
#endif

//#define SABER_X86_TYPE __m128

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberLstmp<X86, OpDtype> :
    public ImplBase <
    X86, OpDtype, LstmParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    //    typedef Tensor<X86> OpTensor;
    SaberLstmp() {}

    ~SaberLstmp() {}

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             LstmParam<X86>& param,
                             Context<X86>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               LstmParam<X86>& param,
                               Context<X86>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 LstmParam<X86>& param);

private:
    LstmParam<X86> _lstm_param;
    Tensor<X86> _lstm_weights;
    Tensor<X86> _gemm_weights;
    Tensor<X86> _inner_output;
    Tensor<X86> _inner_gemm_output;
    SaberLstm<X86, OpDtype> _saber_lstm;
    std::vector<Tensor<X86>*> _inner_ouput_tensor_vec;
    int _output_hidden_size;
    int _inner_hidden_size;

    MklDnnGemm<float, float,float> _wx_gemm_fp32;
    MklDnnGemm<float, float,float> _wh_gemm_fp32;
    MklDnnGemm<float, float,float> _wp_gemm_fp32;

    Tensor<X86> _wx_tensor;
    Tensor<X86> _temp_hidden_tensor;
    Tensor<X86> _temp_cell_tensor;
    int _output_hidden_dim;
    int _inner_hidden_dim;

    Tensor<X86> _inner_x_int8;
    Tensor<X86> _inner_h_int8;
    Tensor<X86> _inner_wh_int32;
    Tensor<X86> _inner_project_scale;
    Tensor<X86> _int8_weights_wx;
    Tensor<X86> _int8_weights_wh;
    Tensor<X86> _int8_weights_project;

    std::vector<float> _inner_scale_wx;
    std::vector<float> _inner_scale_wh;
    std::vector<float> _inner_scale_project;

    MklDnnGemm<int8_t, int8_t, int> _wx_gemm;
    MklDnnGemm<int8_t, int8_t, int> _wh_gemm;
    MklDnnGemm<int8_t, int8_t, int> _wp_gemm;

    PackedFC<AK_INT8, AK_INT8, AK_INT32> _wx_gemm_me;
    PackedFC<AK_INT8, AK_INT8, AK_INT32> _wh_gemm_me;
    PackedFC<AK_INT8, AK_INT8, AK_INT32> _project_gemm_me;

};

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LSTM_H
