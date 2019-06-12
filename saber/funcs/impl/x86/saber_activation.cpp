#include <cmath>
#include <x86intrin.h>
#include "saber/funcs/impl/x86/saber_activation.h"
#include "saber/funcs/impl/x86/saber_normal_activation.h"
#include "mkl.h"
#include "saber/funcs/impl/x86/saber_avx512_funcs.h"
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#include "saber/funcs/debug.h"
#if defined(__AVX2__) and defined(__FMA__)
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#endif


namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberActivation<X86, OpDtype>::init(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ActivationParam<X86>& param,
    Context<X86>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberActivation<X86, OpDtype>::create(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ActivationParam<X86>& param,
    Context<X86>& ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

static void excute_prelu(const std::vector<Tensor<X86>*>& inputs,
                         std::vector<Tensor<X86>*>& outputs,
                         ActivationParam<X86>& param) {
    LayoutType in_layout = inputs[0]->get_layout();
    LayoutType out_layout = outputs[0]->get_layout();
    PreluParam<X86> prelu = param.prelu_param;

#if defined(__AVX2__) and defined(__FMA__)

    if (prelu.channel_shared) {
        for (size_t i = 0; i < inputs.size(); i++) {
            const float* input_data = (float*)inputs[i]->data();
            float* output_data = (float*)outputs[i]->mutable_data();
            int size = inputs[i]->valid_size();
            float* slope_ptr = (float*)prelu.slope->data();
            float alpha = slope_ptr[0];
            const __m256 prelu_alpha = _mm256_set1_ps(alpha);
            int round_length =  size / 8 * 8;
            int remainder = size % 8;

            if (alpha > 1.f) {
                #pragma omp parallel for

                for (int index = 0; index < round_length; index += 8) {
                    __m256 temp = _mm256_load_ps(&input_data[index]);
                    __m256 temp_mul = _mm256_mul_ps(temp, prelu_alpha);
                    temp = _mm256_min_ps(temp, temp_mul);
                    _mm256_store_ps(&output_data[index], temp);
                }

                if (remainder > 0) {
                    __m256i _vec_mask = _m256_continue_mask_m256i(remainder);
                    __m256 temp = _mm256_maskload_ps(&input_data[round_length], _vec_mask);
                    __m256 temp_mul = _mm256_mul_ps(temp, prelu_alpha);
                    __m256  _vec_mask_m256 = _m256_continue_mask_m256(remainder);
                    temp = _mm256_min_ps(temp, temp_mul);
                    _mm256_maskstore_ps(&output_data[round_length], _vec_mask, temp);
                }
            } else {
                #pragma omp parallel for

                for (int index = 0; index < round_length; index += 8) {
                    __m256 temp = _mm256_load_ps(&input_data[index]);
                    __m256 temp_mul = _mm256_mul_ps(temp, prelu_alpha);
                    temp = _mm256_max_ps(temp, temp_mul);
                    _mm256_store_ps(&output_data[index], temp);
                }

                if (remainder > 0) {
                    __m256i _vec_mask = _m256_continue_mask_m256i(remainder);
                    __m256 temp = _mm256_maskload_ps(&input_data[round_length], _vec_mask);
                    __m256 temp_mul = _mm256_mul_ps(temp, prelu_alpha);
                    __m256  _vec_mask_m256 = _m256_continue_mask_m256(remainder);
                    temp = _mm256_max_ps(temp, temp_mul);
                    _mm256_maskstore_ps(&output_data[round_length], _vec_mask, temp);
                }
            }
        }

        return;
    }

#endif


    for (size_t i = 0; i < inputs.size(); i++) {
        const float* input_data = (float*)inputs[i]->data();
        float* output_data = (float*)outputs[i]->mutable_data();
        Shape shin = inputs[i]->valid_shape();
        int num = shin[0];
        int channel = shin[1];
        int size = shin[2] * shin[3];

        for (int n = 0; n < num; n++) {
            const float* in_ptr = input_data + n * channel * size;
            float* out_ptr = output_data + n * channel * size;
            float* slope_ptr = (float*)prelu.slope->data();

            for (int c = 0; c < channel; c++) {
                const float* in_ch_ptr = in_ptr + c * size;
                float* out_ch_ptr = out_ptr + c * size;
                float slope = prelu.channel_shared ?  slope_ptr[0] : slope_ptr[c];

                for (int k = 0; k < size; k++) {
                    out_ch_ptr[k] = in_ch_ptr[k] > 0 ? in_ch_ptr[k] : in_ch_ptr[k] * slope;
                }
            }
        }
    }

}

template <DataType OpDtype>
SaberStatus SaberActivation<X86, OpDtype>::dispatch(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ActivationParam<X86>& param) {
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    // x > 0 ? x :0
    if (param.active == Active_relu) {
        for (size_t vc = 0; vc < inputs.size(); vc++) {
            size_t len = inputs[vc]->valid_size();
            OpDataType* input_data = (OpDataType*)inputs[vc]->mutable_data();
            OpDataType* output_data = (OpDataType*)outputs[vc]->mutable_data();
            outputs[vc]->set_posstive_flag(true);
#if defined(__AVX2__) and defined(__FMA__)
            avx2_vector_relu(input_data, len, output_data);
#else
            #pragma omp parallel for schedule(static)

            for (size_t i = 0; i < len; i++) {
                output_data[i] = input_data[i] > (OpDataType)0 ? input_data[i] : (OpDataType)0;
            }

#endif
        }
    }

    // stanh : b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}
    if (param.active == Active_stanh) {
        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const OpDataType* input_data = (OpDataType*)inputs[i]->data();
            OpDataType* output_data = (OpDataType*)outputs[i]->mutable_data();

            //negative_slope = scale_a
            //coef = scale_b
            for (size_t j = 0; j < len; j++) {
                output_data[j] = param.coef * tanh(param.negative_slope * input_data[j]);
            }
        }
    }

    // sigmoid: 1/(exp(-x) + 1)
    if (param.active == Active_sigmoid) {
        for (size_t i = 0; i < inputs.size() ; i++) {
            size_t len = inputs[i]->valid_size();
            const OpDataType* input_data = (OpDataType*)inputs[i]->data();
            outputs[i]->set_posstive_flag(true);
            OpDataType* output_data = (OpDataType*)outputs[i]->mutable_data();
#if defined(__AVX512F__)
            avx512_vector_sigmoid(input_data, len, output_data);
#elif defined(__AVX2__) and defined(__FMA__)
            avx2_vector_sigmoid(input_data, len, output_data);
#else

            for (size_t j = 0; j < len; j++) {
                output_data[j] = 1.0f / (1.0f + exp(-input_data[j]));
            }

#endif
        }
    }

    // tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    if (param.active == Active_tanh) {

        for (size_t i = 0; i < inputs.size(); i++) {

            size_t len = inputs[i]->valid_size();

            const OpDataType* input_data = (OpDataType*)inputs[i]->data();
            OpDataType* output_data = (OpDataType*)outputs[i]->mutable_data();
            vsTanh(len, input_data, output_data);
        }
    }

    // clipped_relu
    // x > 0 ? x : 0;
    // x < threshold ? x : threshold
    if (param.active == Active_clipped_relu) {
        const OpDataType threshold = param.coef;

        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const OpDataType* input_data = (OpDataType*)inputs[i]->data();
            OpDataType* output_data = (OpDataType*)outputs[i]->mutable_data();
            outputs[i]->set_posstive_flag(true);

            for (size_t j = 0; j < len; j++) {
                output_data[j] = input_data[j] > 0 ? input_data[j] : 0;
                output_data[j] = output_data[j] < threshold ? output_data[j] : threshold;
            }
        }

    }

    //swish: x /(1 + exp(-(b * x)))
    if (param.active == Active_swish) {
        for (size_t i = 0; i < inputs.size(); i++) {
            const OpDataType beta = param.coef;
            size_t len = inputs[i]->valid_size();
            const OpDataType* input_data = (OpDataType*)inputs[i]->data();
            OpDataType* output_data = (OpDataType*)outputs[i]->mutable_data();

            for (size_t j = 0; j < len; j++) {
                output_data[j] = input_data[j] / (1.0f + exp(-input_data[j] * beta));
            }
        }
    }

    //elu:  x > 0 ? x : coef * (exp(x) - 1)
    if (param.active == Active_elu) {
        const OpDataType coef = param.coef;

        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const OpDataType* input_data = (OpDataType*)inputs[i]->data();
            OpDataType* output_data = (OpDataType*)outputs[i]->mutable_data();

            for (size_t j = 0; j < len; j++) {
                output_data[j] = input_data[j] > 0 ? input_data[j] : param.coef * (exp(input_data[j]) - 1);
            }
        }
    }

    //gelu:  y = x * (0.5 * erf(x/sqrt(2)) + 1)
    if (param.active == Active_gelu) {
        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const OpDataType* input_data = (OpDataType*)inputs[i]->data();
            OpDataType* output_data = (OpDataType*)outputs[i]->mutable_data();

            for (size_t j = 0; j < len; j++) {
                OpDataType x  = input_data[j];
                OpDataType coeff = 0.5 * (std::erf(x / sqrt(2)) + 1);

                output_data[j] = x * coeff;
            }
        }
    }

    //prelu: x > 0 ? x : slope[c] * x
    if (param.active == Active_prelu) {
        excute_prelu(inputs, outputs, param);
    }

    for (size_t i = 0; i < inputs.size(); i++) {
        outputs[i]->set_seq_offset(inputs[i]->get_seq_offset());
    }

    return SaberSuccess;
}


template <>
SaberStatus SaberActivation<X86, AK_INT8>::create(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ActivationParam<X86>& param,
    Context<X86>& ctx) {
    this->_ctx = &ctx;

    if (outputs[0]->get_layout() != Layout_NHWC) {
        _output_scale.reshape(
            Shape({outputs[0]->num(), outputs[0]->height(), outputs[0]->width(), outputs[0]->channel()},
                  Layout_NHWC));
    }

    return SaberSuccess;
}
template <>
SaberStatus SaberActivation<X86, AK_INT8>::init(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ActivationParam<X86>& param,
    Context<X86>& ctx) {
    this->_ctx = &ctx;

    for (auto out : outputs) {
        CHECK_EQ(out->get_dtype(), outputs[0]->get_dtype());
    }

    if (outputs[0]->get_layout() != Layout_NHWC) {
        _output_scale.re_alloc(
            Shape({outputs[0]->num(), outputs[0]->height(), outputs[0]->width(), outputs[0]->channel()},
                  Layout_NHWC), outputs[0]->get_dtype());
        _output_scale.set_scale(outputs[0]->get_scale());
    }

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberActivation<X86, AK_INT8>::dispatch(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ActivationParam<X86>& param) {
    typedef typename DataTrait<X86, AK_INT8>::Dtype OpDataType;
    // clipped_relu
    // x > 0 ? x : 0;
    // x < threshold ? x : threshold
    if (param.active == Active_clipped_relu) {
        CHECK(inputs[0]->get_dtype() == AK_INT8);
        CHECK(inputs[0]->get_scale().size() == 1);
        CHECK(outputs[0]->get_scale().size() == 1);
        const float scale_in = inputs[0]->get_scale()[0];
        const float scale_out = inputs[0]->get_scale()[0];
        const float scale_out_rev = 1.f / scale_out;
        const float threshold_fp32 = param.coef;
        outputs[0]->set_scale({scale_in});

        if (outputs[0]->get_dtype() == AK_INT8) {
            for (size_t i = 0; i < inputs.size(); i++) {
                size_t len = inputs[i]->valid_size();
                const OpDataType* input_data = (OpDataType*) inputs[i]->data();
                OpDataType* output_data = (OpDataType*) outputs[i]->mutable_data();

                if (outputs[i]->get_layout() != Layout_NHWC) {
                    output_data = (OpDataType*) _output_scale.mutable_data();
                }

                outputs[i]->set_posstive_flag(true);

                for (size_t j = 0; j < len; j++) {
                    float temp = (float) input_data[j] * scale_in;
                    temp = temp > 0 ? temp : 0;
                    temp = temp < threshold_fp32 ? temp : threshold_fp32;
                    output_data[j] = saturate<int8_t>(roundf(temp * scale_out_rev));
                }

                if (outputs[i]->get_layout() != Layout_NHWC) {
                    reorder_nhwc_nchw(_output_scale, *outputs[i]);
                }
            }
        } else if (outputs[0]->get_dtype() == AK_FLOAT) {
            for (size_t i = 0; i < inputs.size(); i++) {
                size_t len = inputs[i]->valid_size();
                const OpDataType* input_data = (OpDataType*) inputs[i]->data();
                float* output_data = (float*) outputs[i]->mutable_data();

                if (outputs[i]->get_layout() != Layout_NHWC) {
                    output_data = (float*) _output_scale.mutable_data();
                }

                outputs[i]->set_posstive_flag(true);

                for (size_t j = 0; j < len; j++) {
                    float temp = (float) input_data[j] * scale_in;
                    temp = temp > 0 ? temp : 0;
                    output_data[j] = temp < threshold_fp32 ? temp : threshold_fp32;
                }

                if (outputs[i]->get_layout() != Layout_NHWC) {
                    reorder_nhwc_nchw(_output_scale, *outputs[i]);
                }
            }
        } else {
            LOG(FATAL) << "not support ";
        }
    } else {
        LOG(FATAL) << "not support int8 activation " << param.active;
    }

    return SaberSuccess;
}

template class SaberActivation<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberActivation, ActivationParam, X86, AK_HALF);
}
} // namespace anakin
