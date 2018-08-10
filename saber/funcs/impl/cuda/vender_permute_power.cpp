
#include "saber/funcs/impl/cuda/vender_permute_power.h"

namespace anakin{

namespace saber{
template class VenderPermutePower<NV, AK_FLOAT>;

template <>
SaberStatus VenderPermutePower<NV, AK_FLOAT>::\
    create(const std::vector<Tensor<NV>*>&inputs,
           std::vector<Tensor<NV>*>& outputs,
           PermutePowerParam<NV> &param, Context<NV> &ctx) {
        
        if (!(&ctx == this->_ctx)) {
            if (_handle != NULL) {
                CUDNN_CHECK(cudnnDestroy(_handle));
            }
            this->_ctx = &ctx;
            
            cudaStream_t cuda_stream;
            cuda_stream = ctx.get_compute_stream();
            
            CUDNN_CHECK(cudnnCreate(&_handle));
            CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));
        }
        int input_num = inputs[0]->num();
        int input_channel = inputs[0]->channel();
        int input_height = inputs[0]->height();
        int input_width = inputs[0]->width();
        
        bool is_nhwc_to_nchw = param.permute_param.order ==  std::vector<int>({0, 3, 1, 2});
        if (inputs[0]->shape() == inputs[0]->valid_shape()) {
            if (is_nhwc_to_nchw) {
                cudnn::setTensor4dDesc<float>(&_input_descs, CUDNN_TENSOR_NHWC,
                                              input_num, input_width, input_channel, input_height);
                cudnn::setTensor4dDesc<float>(&_output_descs, CUDNN_TENSOR_NCHW,
                                              input_num, input_width, input_channel, input_height);
            } else {
                cudnn::setTensor4dDesc<float>(&_input_descs, CUDNN_TENSOR_NCHW,
                                              input_num, input_channel, input_height, input_width);
                cudnn::setTensor4dDesc<float>(&_output_descs, CUDNN_TENSOR_NHWC,
                                              input_num, input_channel, input_height, input_width);
            }
        } else {
            Shape input_stride = inputs[0]->get_stride();
            Shape output_stride = outputs[0]->get_stride();
            int in_num = inputs[0]->num();
            int in_channel = inputs[0]->channel();
            int in_height = inputs[0]->height();
            int in_width = inputs[0]->width();
            int out_num = outputs[0]->num();
            int out_channel = outputs[0]->channel();
            int out_height = outputs[0]->height();
            int out_width = outputs[0]->width();
            int num_index = inputs[0]->num_index();
            int channel_index = inputs[0]->channel_index();
            int height_index = inputs[0]->height_index();
            int width_index = inputs[0]->width_index();
            if (is_nhwc_to_nchw) {
                cudnn::setTensor4dDescEx<float>(&_input_descs,
                                                in_num, in_width, in_channel, in_height,
                                                input_stride[num_index],
                                                input_stride[width_index],
                                                input_stride[channel_index],
                                                input_stride[height_index]
                                                );
                cudnn::setTensor4dDescEx<float>(&_output_descs,
                                                out_num, out_channel, out_height, out_width,
                                                output_stride[num_index],
                                                output_stride[channel_index],
                                                output_stride[height_index],
                                                output_stride[width_index]
                                                );
            } else {
                cudnn::setTensor4dDescEx<float>(&_input_descs,
                                                in_num, in_channel, in_height, in_width,
                                                input_stride[num_index],
                                                input_stride[channel_index],
                                                input_stride[height_index],
                                                input_stride[width_index]
                                                );
                cudnn::setTensor4dDescEx<float>(&_output_descs,
                                                out_num, out_width, out_channel, out_height,
                                                output_stride[num_index],
                                                output_stride[width_index],
                                                output_stride[channel_index],
                                                output_stride[height_index]
                                                );
            }
        }
        return SaberSuccess;
}

template <>
SaberStatus VenderPermutePower<NV, AK_FLOAT>::\
    init(const std::vector<Tensor<NV> *>& inputs,
                  std::vector<Tensor<NV> *>& outputs,
                  PermutePowerParam<NV> &param, \
                  Context<NV> &ctx) {

        this->_ctx = &ctx;
        // ---- get cuda resources ----

        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();

        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

        // ---- create cudnn Descs ----
        cudnn::createTensorDesc<float>(&_input_descs);
        cudnn::createTensorDesc<float>(&_output_descs);

        return create(inputs, outputs, param, ctx);
}

    //call cudnnConvolutionForward here
template <>
SaberStatus VenderPermutePower<NV, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<NV>*>& inputs,
                          std::vector<Tensor<NV>*>& outputs,
                          PermutePowerParam<NV> &param) {
        const float* input_data = inputs[0]->data();
        float* output_data = outputs[0]->mutable_data();
        float scale = param.power_param.scale;
        float shift = param.power_param.shift;
        float power = param.power_param.power;

        if (shift != 0.f || power != 1.f) {
            LOG(FATAL) << "cudnn permute does not support shift and power";
        } else {
            CUDNN_CHECK(cudnnTransformTensor(_handle,
                                             (void*)(&scale),
                                             _input_descs, input_data,
                                             cudnn::cudnnTypeWrapper<float>::kZero(),
                                             _output_descs, output_data));
        }

        return SaberSuccess;
}

} //namespace saber

} //namespace anakin
