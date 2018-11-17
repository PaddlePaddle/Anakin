/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_ACTIVATION_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_ACTIVATION_H
#include "saber/funcs/impl/impl_activation.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
namespace anakin {

namespace saber {

template <DataType OpDtype>
class VenderActivation<NV, OpDtype> : public ImplBase<
        NV, OpDtype, ActivationParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    typedef typename DataTrait<NV, OpDtype>::Dtype InDataType;
    typedef typename DataTrait<NV, OpDtype>::Dtype OutDataType;

    VenderActivation()
            : _handle(NULL), _active_descs(NULL), _input_descs(NULL), _output_descs(NULL) {}

    ~VenderActivation() {
        if (_input_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_input_descs));
        }
        if (_output_descs) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(_output_descs));
        }
        if (_active_descs != NULL) {
            CUDNN_CHECK(cudnnDestroyActivationDescriptor(_active_descs));
        } 
		if (_handle != NULL) { 
			CUDNN_CHECK(cudnnDestroy(_handle)); 
		}
    }

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            ActivationParam<NV>& param, Context<NV>& ctx) {

        this->_ctx = &ctx;

        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();

        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));

        cudnn::createTensorDesc<InDataType>(&_input_descs);
        cudnn::createTensorDesc<OutDataType>(&_output_descs);

        cudnn::create_activation_des<OpDataType>(&_active_descs);

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            ActivationParam<NV>& param, Context<NV>& ctx) {
        if (param.active == Active_prelu || param.active == Active_stanh) {
            return SaberUnImplError;
        }
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
        int output_channel = outputs[0]->channel();
        int output_height = outputs[0]->height();
        int output_width = outputs[0]->width();

        Shape in_stride = inputs[0]->get_stride();
        Shape out_stride = outputs[0]->get_stride();

        int dim_a[] = {input_num, input_channel,
                       input_height, input_width};

        int dim_b[] = {input_num, output_channel,
                       output_height, output_width};

        cudnn::setTensorNdDesc<InDataType>(&_input_descs,
                                          inputs[0]->dims(), dim_a, &in_stride[0]);

        cudnn::setTensorNdDesc<OutDataType>(&_output_descs,
                                          outputs[0]->dims(), dim_b, &out_stride[0]);

        cudnn::set_activation_des<OpDataType>(&_active_descs, param.active, param.coef);

        return SaberSuccess;
    }

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            ActivationParam<NV>& param) {

        if (param.active == Active_prelu || param.active == Active_stanh) {
            return SaberUnImplError;
        }
        const InDataType *in_data = (const InDataType *) inputs[0]->data();
        OutDataType *out_data = (OutDataType *) outputs[0]->mutable_data();

        CUDNN_CHECK(cudnnActivationForward(_handle, _active_descs,
                                           cudnn::cudnnTypeWrapper<InDataType>::kOne(),
                                           _input_descs, in_data,
                                           cudnn::cudnnTypeWrapper<InDataType>::kZero(),
                                           _output_descs, out_data
        ));
        return SaberSuccess;
    }

private:
    cudnnHandle_t _handle;
    cudnnTensorDescriptor_t _input_descs;
    cudnnTensorDescriptor_t _output_descs;
    cudnnActivationDescriptor_t _active_descs;
};
template class VenderActivation<NV, AK_FLOAT>;
}
}

#endif //ANAKIN_SABER_FUNCS_VENDER_ACTIVATION_H
