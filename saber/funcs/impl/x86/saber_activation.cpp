
#include "saber/funcs/impl/x86/saber_activation.h"
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberActivation<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ActivationParam<NV> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberActivation<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ActivationParam<NV> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberActivation<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ActivationParam<NV> &param) {
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    typedef typename DataTrait<NV, OpDtype>::Dtype DataType_in;
    typedef typename DataTrait<NV, OpDtype>::Dtype DataType_out;
    // TODO !! need add other types of activation
    if (param.active == Active_relu) {
        for (size_t vc = 0; vc < inputs.size(); vc++) {
            size_t len = inputs[vc]->valid_size();
            float *input_data = (float*)inputs[vc]->mutable_data();
            float *output_data = (float*)outputs[vc]->mutable_data();

            for (size_t i = 0; i < len; i++) {
                if (*input_data > 0) {
                    *output_data = *input_data;
                } else {
                    *output_data = 0;
                }

                input_data++;
                output_data++;
            }
        }
    }

    // stanh : b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}
    if (param.active == Active_stanh) {
        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const DataType_in *input_data = (float*)inputs[i]->data();
            DataType_out *output_data = (float*)outputs[i]->mutable_data();
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
            const DataType_in *input_data = (float*)inputs[i]->data();
            DataType_out *output_data = (float*)outputs[i]->mutable_data();

            for (size_t j = 0; j < len; j++) {
                output_data[j] = 1.0f / (1.0f + exp(-input_data[j]));
            }
        }
    }

    // tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    if (param.active == Active_tanh) {
        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const DataType_in *input_data = (float*)inputs[i]->data();
            DataType_out *output_data = (float*)outputs[i]->mutable_data();

            for (size_t j = 0; j < len; j++) {
                output_data[j] = tanh(input_data[j]);
            }
        }
    }

    // clipped_relu
    // x > 0 ? x : 0;
    // x < threshold ? x : threshold
    if (param.active == Active_clipped_relu) {
        const DataType_in threshold = param.coef;
        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const DataType_in *input_data = (float*)inputs[i]->data();
            DataType_out *output_data = (float*)outputs[i]->mutable_data();

            for(size_t j = 0; j < len; j++){
                output_data[j] = input_data[j] > 0 ? input_data[j] : 0;
                output_data[j] = output_data[j] < threshold ? output_data[j] : threshold;
            }
        }
    }

    //elu:  x > 0 ? x : coef * (exp(x) - 1)
    if (param.active == Active_elu) {
        const DataType_in coef = param.coef;
        for (size_t i = 0; i < inputs.size(); i++) {
            size_t len = inputs[i]->valid_size();
            const DataType_in *input_data = (float*)inputs[i]->data();
            DataType_out *output_data = (float*)outputs[i]->mutable_data();

            for(size_t j = 0; j < len; j++){
                output_data[j] = input_data[j] > 0 ? input_data[j] : param.coef * (exp(input_data[j]) - 1);
            }
        }
    }
    for (size_t i = 0; i < inputs.size(); i++) {
        outputs[i]->set_seq_offset(inputs[i]->get_seq_offset());
    }
    return SaberSuccess;
}

template class SaberActivation<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}
} // namespace anakin
