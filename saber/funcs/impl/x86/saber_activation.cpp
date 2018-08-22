
#include "saber/funcs/impl/x86/saber_activation.h"
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberActivation<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ActivationParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberActivation<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ActivationParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberActivation<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ActivationParam<X86> &param) {
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    // x > 0 ? x :0
    if (param.active == Active_relu) {
        for (size_t vc = 0; vc < inputs.size(); vc++) {
            size_t len = inputs[vc]->valid_size();
            OpDataType *input_data = (OpDataType*)inputs[vc]->mutable_data();
            OpDataType *output_data = (OpDataType*)outputs[vc]->mutable_data();

            for (size_t i = 0; i < len; i++) {
                *output_data = *input_data > (OpDataType)0 ? *input_data : (OpDataType)0;
                input_data++;
                output_data++;
            }
        }
    }

    // stanh : b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}
    if (param.active == Active_stanh) {
        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const OpDataType *input_data = (OpDataType*)inputs[i]->data();
            OpDataType *output_data = (OpDataType*)outputs[i]->mutable_data();
            //negative_slope = scale_a
            //coef = scale_b
            for (size_t j = 0; j < len; j++) {
                output_data[j] = param.coef * tanh(param.negative_slope * input_data[j]);
            }
        }
    }
    // sigmoid: 1/(exp(-x) + 1)
    if (param.active == Active_sigmoid) {
        for ( size_t i = 0; i < inputs.size() ; i++) {
            size_t len = inputs[i]->valid_size();
            const OpDataType *input_data = (OpDataType*)inputs[i]->data();
            OpDataType *output_data = (OpDataType*)outputs[i]->mutable_data();

            for (size_t j = 0; j < len; j++) {
                output_data[j] = 1.0f / (1.0f + exp(-input_data[j]));
            }
        }
    }

    // tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    if (param.active == Active_tanh) {
        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const OpDataType *input_data = (OpDataType*)inputs[i]->data();
            OpDataType *output_data = (OpDataType*)outputs[i]->mutable_data();

            for (size_t j = 0; j < len; j++) {
                output_data[j] = tanh(input_data[j]);
            }
        }
    }

    // clipped_relu
    // x > 0 ? x : 0;
    // x < threshold ? x : threshold
    if (param.active == Active_clipped_relu) {
        const OpDataType threshold = param.coef;
        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const OpDataType *input_data = (OpDataType*)inputs[i]->data();
            OpDataType *output_data = (OpDataType*)outputs[i]->mutable_data();

            for(size_t j = 0; j < len; j++){
                output_data[j] = input_data[j] > 0 ? input_data[j] : 0;
                output_data[j] = output_data[j] < threshold ? output_data[j] : threshold;
            }
        }
    }

    //elu:  x > 0 ? x : coef * (exp(x) - 1)
    if (param.active == Active_elu) {
        const OpDataType coef = param.coef;
        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const OpDataType *input_data = (OpDataType*)inputs[i]->data();
            OpDataType *output_data = (OpDataType*)outputs[i]->mutable_data();

            for(size_t j = 0; j < len; j++){
                output_data[j] = input_data[j] > 0 ? input_data[j] : param.coef * (exp(input_data[j]) - 1);
            }
        }
    }
    //prelu: x > 0 ? x : slope[c] * x
    if (param.active == Active_prelu) {
        PreluParam<X86> prelu = param.prelu_param;
        for (size_t i = 0; i < inputs.size(); i++) {
            const OpDataType *input_data = (OpDataType*)inputs[i]->data();
            OpDataType *output_data = (OpDataType*)outputs[i]->mutable_data();
            Shape shin = inputs[i]->valid_shape();
            int num = shin[0];
            int channel = shin[1];
            int size = shin[2] * shin[3];
            for (int n = 0; n < num; n++){
                const OpDataType *in_ptr = input_data + n * channel * size;
                OpDataType *out_ptr = output_data + n * channel * size;
                OpDataType *slope_ptr = (OpDataType*)prelu.slope->data();
                for (int c = 0; c < channel; c++){
                    const OpDataType *in_ch_ptr = in_ptr + c * size;
                    OpDataType *out_ch_ptr = out_ptr + c * size;
                    OpDataType slope = prelu.channel_shared ?  slope_ptr[0]: slope_ptr[c];
                    for (int k = 0; k < size; k++){
                        out_ch_ptr[k] = in_ch_ptr[k] > 0 ? in_ch_ptr[k] : in_ch_ptr[k] * slope;
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < inputs.size(); i++) {
        outputs[i]->set_seq_offset(inputs[i]->get_seq_offset());
    }
    return SaberSuccess;
}

template class SaberActivation<X86, AK_FLOAT>;

}
} // namespace anakin
