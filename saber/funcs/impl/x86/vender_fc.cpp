#include "saber/funcs/impl/x86/vender_fc.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "mkl_cblas.h"
#include "mkl_vml_functions.h"
#include "tensor_op.h"

namespace anakin {
namespace saber {

typedef MKL_INT cblas_int;

template <>
void VenderFc<X86, AK_FLOAT>::clean() {
    if (bias_sum) {
        free(bias_sum);
        bias_sum = nullptr;
    }

    for (int i = packed_weights.size() - 1; i >= 0; i--) {
        float* pw = packed_weights[i];
        cblas_sgemm_free(pw);
        pw = nullptr;
        packed_weights.pop_back();
    }

    std::vector<OpDataType*>().swap(packed_weights);
}



template <>
SaberStatus VenderFc<X86, AK_FLOAT>
::create(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         FcParam<X86>& param, Context<X86>& ctx) {

    this->_ctx = &ctx;
    this->_param = &param;

    MB = inputs[0]->count_valid(0, param.axis);
    OC = outputs[0]->channel();

    // weights
    for (int i = packed_weights.size() - 1; i >= 0; i--) {
        cblas_sgemm_free(packed_weights[i]);
    }

    std::vector<float*> ().swap(packed_weights);

    const float* weights = (const float*)param.weights->data();

    if (_need_weights_trans) {
        weights = static_cast<const float*>(_weights_trans.data());
    }

    int total_IC = 0;

    for (int i = 0; i < inputs.size(); i++) {
        cblas_int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());
        packed_weights.push_back(cblas_sgemm_alloc(CblasAMatrix, OC, MB, IC));
        // LOG(INFO) << "anakin input[" << i << "] alloc passed";
        cblas_sgemm_pack(CblasColMajor,
                         CblasAMatrix,
                         param.is_transpose_weights ? CblasNoTrans : CblasTrans,
                         OC, MB, IC,
                         1.0,
                         weights + total_IC * OC, IC,
                         packed_weights[i]);
        total_IC += IC;
        // LOG(INFO) << "anakin input[" << i << "] pack passed";
    }

    CHECK_EQ(inputs.size(), 1);

    if (inputs[0]->get_dtype() != AK_FLOAT) {
        utils::try_expand_tensor(_input_scale, inputs[0]->valid_shape());
    }

    return SaberSuccess;
}

template <>
SaberStatus VenderFc<X86, AK_FLOAT>
::init(const std::vector<Tensor<X86> *>& inputs,
       std::vector<Tensor<X86> *>& outputs,
       FcParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    LayoutType in_layout = inputs[0]->get_layout();
    LayoutType out_layout = outputs[0]->get_layout();

    if (in_layout == Layout_NCHW_C8R && out_layout == Layout_NCHW) {
        CHECK(inputs[0]->channel() % 8 == 0) << "only support channel div 8 == 0";
        _need_weights_trans = true;
        _weights_trans.re_alloc(param.weights->valid_shape());
        int oc_value = param.weights->height();
        int oc_stride = param.weights->width();
        int ic_value = inputs[0]->channel();
        int c_value_div_8 = ic_value / 8;
        int hw_value = inputs[0]->height() * inputs[0]->width();
        float* out_weights = static_cast<float*>(_weights_trans.mutable_data());
        const float* in_weights = static_cast<const float*>(param.weights->data());

        for (int oc = 0; oc < oc_value; oc++) {
            for (int ic_div_8 = 0; ic_div_8 < c_value_div_8; ic_div_8++) {
                for (int hw = 0; hw < hw_value; hw++) {
                    for (int inner_c = 0; inner_c < 8; inner_c++) {
                        int out_index = oc * oc_stride + ic_div_8 * hw_value * 8 + hw * 8 + inner_c;
                        int in_index = oc * oc_stride + (ic_div_8 * 8 + inner_c) * hw_value + hw;
                        out_weights[out_index] = in_weights[in_index];
                    }
                }
            }
        }

        DLOG(INFO) << "ak trans weights nchw  to c8r";
    } else if (in_layout == Layout_NHWC && out_layout == Layout_NCHW) {
        _need_weights_trans = true;
        _weights_trans.re_alloc(param.weights->valid_shape());
        int oc_value = param.weights->height();
        int oc_stride = param.weights->width();
        int ic_value = inputs[0]->channel();
        int hw_value = inputs[0]->height() * inputs[0]->width();
        float* out_weights = static_cast<float*>(_weights_trans.mutable_data());
        const float* in_weights = static_cast<const float*>(param.weights->data());

        for (int oc = 0; oc < oc_value; oc++) {
            for (int hw = 0; hw < hw_value; hw++) {
                for (int ic = 0; ic < ic_value; ic++) {
                    int out_index = oc * oc_stride + hw * ic_value + ic;
                    int in_index = oc * oc_stride + ic * hw_value + hw;
                    out_weights[out_index] = in_weights[in_index];
                }
            }
        }

        DLOG(INFO) << "ak trans weights nchw to nchwc";
    } else if ((in_layout == Layout_NCHW || in_layout == Layout_NC || in_layout == Layout_NHW
                || in_layout == Layout_HW)
               && out_layout == Layout_NCHW) {
        _need_weights_trans = false;
    } else {
        LOG(FATAL) << "not support input layout in = " << inputs[0]->get_layout() << " , out = " <<
                   outputs[0]->get_layout();
    }

    CHECK_EQ(inputs.size(), 1);

    if (inputs[0]->get_dtype() != AK_FLOAT) {
        _input_scale.re_alloc(inputs[0]->valid_shape(), AK_FLOAT);
    }

    return create(inputs, outputs, param, ctx);
}


template <>
SaberStatus VenderFc<X86, AK_FLOAT>
::dispatch(const std::vector<Tensor<X86> *>& inputs,
           std::vector<Tensor<X86> *>& outputs,
           FcParam<X86>& param) {

    float* dst = (float*)outputs[0]->mutable_data();
    const float* bias = NULL;

    if (param.bias) {
        bias = (const float*)param.bias->data();
    }

    for (int i = 0; i < inputs.size(); i++) {

        const float* src = nullptr;

        if (inputs[i]->get_dtype() == AK_FLOAT) {
            src = static_cast<const float*>(inputs[i]->data());
        } else if (inputs[i]->get_dtype() == AK_UINT8) {
            DLOG(INFO) << "dispatch convert uint8 fp32";
            utils::ScaleUtils::scale_uint8_fp32(_input_scale, *inputs[i]);
            src = static_cast<const float*>(_input_scale.data());
        }


        cblas_int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());

        if (i == 0) {
            // C := alpha * op(A) * op(B) + beta * C
            cblas_sgemm_compute(CblasColMajor,                                     // Layout
                                CblasPacked,                                       // a
                                CblasNoTrans,                                      // b是否转置
                                OC, MB, IC,                                        // m, n, k
                                packed_weights[i], IC,                             // a, lda
                                src, IC,                                           // b, ldb
                                0.0,                                               // beta
                                dst, OC);                                          // c, ldc
        } else {
            cblas_sgemm_compute(CblasColMajor,                                     // Layout
                                CblasPacked,                                       // a
                                CblasNoTrans,                                      // b是否转置
                                OC, MB, IC,                                        // m, n, k
                                packed_weights[i], IC,                             // a, lda
                                src, IC,                                           // b, ldb
                                1.0,                                               // beta
                                dst, OC);                                          // c, ldc
        }

        //LOG(INFO) << "anakin compute[" << i << "] passed";

        // LOG(INFO) << "inputs[]:dims: " << inputs[0]->dims();
        // LOG(INFO) << "inputs:size: " << inputs.size();
        // LOG(INFO) << "inputs:capacity: " << inputs.capacity();
        // LOG(INFO) << "output:size: " << outputs.size();
        // LOG(INFO) << "OC, MB, IC: " << OC << " "<< MB << " " << IC;
    }

    if (bias) {
        #pragma omp parallel for schedule(static)

        for (cblas_int mb = 0; mb < MB; mb++) {
            cblas_saxpy(OC, 1.0, bias, 1.0, dst + mb * OC, 1);
        }
    }

    return SaberSuccess;
}
template class VenderFc<X86, AK_FLOAT>;


template <>
void VenderFc<X86, AK_INT8>::clean() {
    if (ws_) {
        zfree(ws_);
        ws_ = nullptr;
    }
}

template <>
SaberStatus VenderFc<X86, AK_INT8>::create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        FcParam<X86>& param,
        Context<X86>& ctx) {
    if (inputs[0]->get_dtype() == AK_INT8 || inputs[0]->get_dtype() == AK_FLOAT) {
        return SaberSuccess;
    }

    if (ws_) {
        zfree(ws_);
        ws_ = nullptr;
    }

    //    LOG(INFO)<<"batch size = "<<_batch_size<<","<<_output_channel;
    ws_ = zmalloc(_batch_size * _output_channel * sizeof(int), 256);

    if (ws_ == nullptr) {
        LOG(FATAL) << "OutOfMem";
        return SaberOutOfMem;
    }

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        utils::try_expand_tensor(_input_scale, inputs[0]->valid_shape());
    }

    return SaberSuccess;
}

template <>
SaberStatus VenderFc<X86, AK_INT8>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        FcParam<X86>& param,
        Context<X86>& ctx) {
    if (inputs[0]->get_dtype() == AK_INT8 || inputs[0]->get_dtype() == AK_FLOAT) {
        int m = inputs[0]->count_valid(0, param.axis);
        int n = outputs[0]->channel();
        int k = inputs[0]->count_valid(param.axis, inputs[0]->dims());
        CHECK(inputs[0]->get_scale().size() > 0);

        _packed_int8_gemm.init(false, true, m, n, k, *param.weights, inputs[0]->get_scale()[0]);
        return SaberSuccess;
    }

    this->_ctx = &ctx;
    this->_param = &param;

    CHECK(inputs[0]->get_dtype() == AK_FLOAT
          || inputs[0]->get_dtype() == AK_UINT8) << "not support input type " << inputs[0]->get_dtype();
    CHECK_GT(inputs[0]->get_scale().size(), 0) << "input scale must >0";
    CHECK_GT(outputs[0]->get_scale().size(), 0) << "output scale must >0";

    _output_channel = outputs[0]->channel();
    _batch_size = inputs[0]->count_valid(0, param.axis);

    if (param.weights->get_dtype() == AK_FLOAT) {
        _need_weights_trans = true;
        _weights_trans.re_alloc(param.weights->valid_shape(), AK_INT8);
        utils::ScaleUtils::scale_fc_weights_to_nchw_host(_weights_trans, *param.weights);
        //        LOG(INFO)<<"input shape "<<inputs[0]->valid_shape()<<" , weights shape "<<param.weights->valid_shape();
    }

    if (_need_weights_trans) {
        for (int i = 0; i < _output_channel; i ++) {
            _scale.push_back((inputs[0]->get_scale()[0] * _weights_trans.get_scale()[i]) /
                             outputs[0]->get_scale()[0]);
        }
    } else {
        for (int i = 0; i < _output_channel; i ++) {
            _scale.push_back((inputs[0]->get_scale()[0] * param.weights->get_scale()[i]) /
                             outputs[0]->get_scale()[0]);
        }
    }

    if (param.bias != nullptr && param.bias->valid_size() > 0 && param.bias->get_dtype() == AK_FLOAT) {
        _bias_scale.re_alloc(param.bias->valid_shape(), AK_INT32);
        _bias_scale.set_scale(_scale);
        utils::ScaleUtils::scale_bias_fp32_int32(_bias_scale, *param.bias);
    }

    _is_transpose_weights = param.is_transpose_weights ?
                            CblasNoTrans :
                            CblasTrans;

    if (inputs[0]->get_dtype() == AK_FLOAT) {
        _input_scale.re_alloc(inputs[0]->valid_shape(), AK_UINT8);
    }

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderFc<X86, AK_INT8>::dispatch(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        FcParam<X86>& param) {

    if (inputs[0]->get_dtype() == AK_INT8 || inputs[0]->get_dtype() == AK_FLOAT) {
        int m = inputs[0]->count_valid(0, param.axis);
        _packed_int8_gemm.dispatch(1.f, 0.f, m, *inputs[0], *outputs[0], param.bias);
        return SaberSuccess;
    }

#define __FC_PARALLEL_FUNC [&](int mb, int oc) { \
    int dst_index = mb * _output_channel + oc; \
    if (bias) { \
        dst[dst_index] = (_scale[oc] == 1.f) ? \
            static_cast<int32_t *>(ws_)[dst_index] + bias[oc] : \
            _scale[oc] * (static_cast<int32_t *>(ws_)[dst_index] + bias[oc]); \
    } else { \
        dst[dst_index] = (_scale[oc] == 1.f) ? \
            dst[dst_index] = static_cast<int32_t *>(ws_)[dst_index] : \
            _scale[oc] * static_cast<int32_t *>(ws_)[dst_index]; \
    } \
}

    int c_offset = 0;
    int total_ic = 0;

    auto bias = param.bias != nullptr && param.bias->valid_size() > 0 ?
                (param.bias->get_dtype() == AK_INT32 ? static_cast<const int*>(param.bias->data()) :
                 static_cast<const int*>(_bias_scale.data()))
                : nullptr;

    for (int i = 0; i < inputs.size(); i++) {
        int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());

        auto src = static_cast<const uint8_t*>(inputs[i]->data());

        if (inputs[i]->get_dtype() == AK_FLOAT) {
            utils::ScaleUtils::scale_fp32_uint8(_input_scale, *inputs[0]);
            src = static_cast<const uint8_t*>(_input_scale.data());
            //            print_tensor(_input_scale);
        }

        auto weight = static_cast<const int8_t*>(param.weights->data()) + total_ic * _output_channel;

        if (_need_weights_trans) {
            //            LOG(INFO)<<"weights trans";
            weight = static_cast<const int8_t*>(_weights_trans.data()) + total_ic * _output_channel;
            //            print_tensor(_weights_trans);
        }

        //        for(auto a:_scale){
        //            LOG(INFO)<<"scale = "<<a;
        //        }
        //        LOG(INFO)<<"m,n,k = "<<_output_channel<<","<<_batch_size<<","<<IC;
        //        print_tensor(_bias_scale);
        /* c = scale * { op(A) + a_offset_scale * a_offset } *
               { op(B) + b_offset_scale * b_offset } + beta * C + c_offset */
        if (i == 0) {
            cblas_gemm_s8u8s32(CblasColMajor,                       // Layout
                               _is_transpose_weights,                // a need to transpose or not
                               CblasNoTrans,                        // b need to transpose or not
                               CblasFixOffset,                      // c_offset_layout
                               _output_channel,                      // m
                               _batch_size,                          // n
                               IC,                                  // k
                               1.0,                                 // scale
                               weight,                              // a
                               IC,                                  // lda
                               0,                                   // a_offset
                               src,                                 // b
                               IC,                                  // ldb
                               0,                                   // b_offset
                               0.0,                                 // beta
                               static_cast<int*>(ws_),              // c
                               _output_channel,                      // ldc
                               &c_offset);
        } else {
            cblas_gemm_s8u8s32(CblasColMajor,
                               _is_transpose_weights,
                               CblasNoTrans,
                               CblasFixOffset,
                               _output_channel,
                               _batch_size,
                               IC,
                               1.0,
                               weight,
                               IC,
                               0,
                               src,
                               IC,
                               0,
                               1.0,
                               static_cast<int*>(ws_),
                               _output_channel,
                               &c_offset);
        }

        total_ic += IC;
    }

    auto dst_dtype = outputs[0]->get_dtype();

    if (dst_dtype == AK_FLOAT) {
        auto dst = static_cast<float*>(outputs[0]->mutable_data());
        parallel_nd(_batch_size, _output_channel, __FC_PARALLEL_FUNC);
    } else if (dst_dtype == AK_INT32) {
        auto dst = static_cast<int32_t*>(outputs[0]->mutable_data());
        parallel_nd(_batch_size, _output_channel, __FC_PARALLEL_FUNC);
    } else if (dst_dtype == AK_INT8) {
        auto dst = static_cast<int8_t*>(outputs[0]->mutable_data());
        parallel_nd(_batch_size, _output_channel, __FC_PARALLEL_FUNC);
    } else {
        LOG(FATAL) << "not support this type " << dst_dtype;
        return SaberUnImplError;
    }

    return SaberSuccess;
}

template class VenderFc<X86, AK_INT8>;

DEFINE_OP_TEMPLATE(VenderFc, FcParam, X86, AK_HALF);

} // namespace saber
} // namespace anakin
