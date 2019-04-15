#include "saber/funcs/impl/x86/saber_conv_1x1.h"
#include "mkl_cblas.h"
#include "saber/funcs/timer.h"

namespace anakin {
namespace saber {
//inline
static inline void gemm(const bool trans_a, const bool transb, int m, int n, int k,
                        const float alpha,
                        const float* a, const float* b, const float beta, float* c) {
    //    cout << "(" << m << "," << n << "," << k << ")" << endl;
    int lda = (!trans_a/* == CblasNoTrans*/) ? k : m;
    int ldb = (!transb/* == CblasNoTrans*/) ? n : k;
    CBLAS_TRANSPOSE cblas_transa =
        (!trans_a/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cblas_transb =
        (!transb/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    //    LOG(INFO)<<"m "<<m<<","<<n<<","<<k<<","<<beta;
    cblas_sgemm(CblasRowMajor, cblas_transa, cblas_transb, m, n, k, alpha, a, k, b, n, beta, c, n);
};

template <>
SaberStatus SaberConv1X1<AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    ConvParam<X86>* conv_param = &param.conv_param;
    _out_c = conv_param->weight()->num();
    _in_c = conv_param->weight()->channel();
    int h = inputs[0]->height();
    int w = inputs[0]->width();
    _in_inner_size = h * w;
    _num_input = inputs[0]->num();
    _num_size_in = _in_c * h * w;
    _num_size_out = _out_c * h * w;

    _add_output = 0.f;

    if (param.eltwise_param.has_eltwise) {
        _add_output = 1.f;
    }

    DLOG(INFO) << "flag :" << _flag_bias << "," << _flag_relu << "," << _flag_neg;
    return SaberSuccess;
}

template <>
SaberStatus SaberConv1X1<AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    ConvParam<X86>* conv_param = &param.conv_param;
    EltwiseParam<X86>* elt_param = &param.eltwise_param;
    _flag_bias = (conv_param->bias() != nullptr) && (conv_param->bias()->valid_size() > 0);

    if (conv_param->activation_param.active == Active_relu) {
        _flag_relu = true;
        _flag_neg = conv_param->activation_param.negative_slope != 0.f;
        _neg_slope = conv_param->activation_param.negative_slope;
    } else if (elt_param->activation_param.active == Active_relu) {
        _flag_relu = true;
        _flag_neg = elt_param->activation_param.negative_slope != 0.f;
        _neg_slope = elt_param->activation_param.negative_slope;
    } else {
        _flag_relu = false;
        _flag_neg = false;
        _neg_slope = 0.f;
    }

    _bias_utils.reset(_flag_bias, _flag_relu, _flag_neg);



    return create(inputs, outputs, param, ctx);

}

template <>
SaberStatus SaberConv1X1<AK_FLOAT>::dispatch(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param) {

    ConvParam<X86>* conv_param = &param.conv_param;
    const float* weights_data = static_cast<const float* >(conv_param->weight()->data());
    const float* in_data = static_cast<const float*>(inputs[0]->data());
    float* out_data = static_cast<float*>(outputs[0]->mutable_data());


    //    SaberTimer<X86> timer;
    //    timer.start(*this->_ctx);
    for (int batch_id = 0; batch_id < inputs[0]->num(); batch_id++) {
        gemm(false, false, _out_c, _in_inner_size, _in_c, 1.f, weights_data,
             &in_data[0 + batch_id * _in_c * _in_inner_size], _add_output,
             &out_data[0 + batch_id * _out_c * _in_inner_size]);
    }

    //    timer.end(*this->_ctx);
    //    double use_ms=timer.get_average_ms();
    //    double work_load=(double)_out_c*_in_inner_size*_in_c*2;
    //    double speed=work_load/use_ms/1000.0/1000.0;
    //    LOG(INFO)<<"speed "<<speed<<",time  = "<<use_ms;

    //        LOG(INFO)<<"it is me";

    if (_flag_bias) {
        _bias = static_cast<const float*>(conv_param->bias()->data());
    }

    _bias_utils.run(out_data, _bias, _num_input, _out_c, _in_inner_size, _neg_slope);

    return SaberSuccess;
}

}
}
