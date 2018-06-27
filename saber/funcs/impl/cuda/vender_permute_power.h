/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_PERMUTE_POWER_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_PERMUTE_POWER_H

#include "saber/funcs/impl/impl_permute_power.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class VenderPermutePower<NV, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
    Tensor<NV, inDtype, LayOutType_in>,
    Tensor<NV, outDtype, LayOutType_out>,
    Tensor<NV, OpDtype, LayOutType_op>,
    PermutePowerParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderPermutePower()
            : _handle(NULL)
            , _input_descs(NULL)
            , _output_descs(NULL)
    {}

    ~VenderPermutePower() {

        if (_input_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_descs));
        }
        if (_output_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_descs));
        }
        if (_handle != NULL) {
            CUDNN_CHECK(cudnnDestroy(_handle));
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  PermutePowerParam<OpTensor> &param, \
                  Context<NV> &ctx) {

        this->_ctx = &ctx;
        // ---- get cuda resources ----

        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();

        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

        // ---- create cudnn Descs ----
        cudnn::createTensorDesc<InDataType>(&_input_descs);
        cudnn::createTensorDesc<OutDataType>(&_output_descs);

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>&inputs,
                std::vector<DataTensor_out*>& outputs,
                PermutePowerParam<OpTensor> &param, Context<NV> &ctx) {

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
                cudnn::setTensor4dDesc<InDataType>(&_input_descs, CUDNN_TENSOR_NHWC,
                        input_num, input_width, input_channel, input_height);
                cudnn::setTensor4dDesc<OutDataType>(&_output_descs, CUDNN_TENSOR_NCHW,
                        input_num, input_width, input_channel, input_height);
            } else {
                cudnn::setTensor4dDesc<InDataType>(&_input_descs, CUDNN_TENSOR_NCHW,
                        input_num, input_channel, input_height, input_width);
                cudnn::setTensor4dDesc<OutDataType>(&_output_descs, CUDNN_TENSOR_NHWC,
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
                cudnn::setTensor4dDescEx<InDataType>(&_input_descs,
                    in_num, in_width, in_channel, in_height,
                    input_stride[num_index],
                    input_stride[width_index],
                    input_stride[channel_index],
                    input_stride[height_index]
                    );
                cudnn::setTensor4dDescEx<OutDataType>(&_output_descs,
                    out_num, out_channel, out_height, out_width,
                    output_stride[num_index],
                    output_stride[channel_index],
                    output_stride[height_index],
                    output_stride[width_index]
                    );
            } else {
                cudnn::setTensor4dDescEx<InDataType>(&_input_descs,
                    in_num, in_channel, in_height, in_width,
                    input_stride[num_index],
                    input_stride[channel_index],
                    input_stride[height_index],
                    input_stride[width_index]
                    );
                cudnn::setTensor4dDescEx<OutDataType>(&_output_descs,
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
    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          PermutePowerParam<OpTensor> &param) {
        const InDataType* input_data = inputs[0]->data();
        OutDataType* output_data = outputs[0]->mutable_data();
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
private:
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _input_descs;
    cudnnTensorDescriptor_t _output_descs;
    const bool _use_tensor_core = true;
};
template class VenderPermutePower<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_CUDNN_PERMUTE_POWER_H
