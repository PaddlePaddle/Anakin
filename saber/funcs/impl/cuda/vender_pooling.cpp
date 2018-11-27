
#include "saber/funcs/impl/cuda/vender_pooling.h"

#include "saber/funcs/impl/impl_pooling.h"

namespace anakin{
namespace saber {
    
template class VenderPooling<NV, AK_FLOAT>;
    
template <>
SaberStatus VenderPooling<NV, AK_FLOAT>::\
    create(const std::vector<DataTensor_in*>& inputs,
           std::vector<DataTensor_out*>& outputs,
           PoolingParam<NV> &pooling_param, Context<NV> &ctx) {
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
        
        Shape stride_in = inputs[0]->get_stride();
        Shape stride_out = outputs[0]->get_stride();
        
        int dim_a[] = {input_num, input_channel,
            input_height, input_width};
        
        int dim_b[] = {input_num, output_channel,
            output_height, output_width};
        
        cudnn::setTensorNdDesc<float>(&_input_descs,
                                      inputs[0]->dims(), dim_a, &stride_in[0]);
        
        cudnn::setTensorNdDesc<float>(&_output_descs,
                                      outputs[0]->dims(), dim_b, &stride_out[0]);
        
        int windowHeight[] = {pooling_param.window_h, pooling_param.window_w};
        int padding[] = {pooling_param.pad_h, pooling_param.pad_w};
        
        int stride[] = {pooling_param.stride_h, pooling_param.stride_w};
        
        cudnn::set_nd_pooling_des<float>(&_pooling_descs, pooling_param.pooling_type,
                                         inputs[0]->dims() - 2, windowHeight,
                                         padding,stride);
        return SaberSuccess;
}
    
template<>
SaberStatus VenderPooling<NV, AK_FLOAT> :: \
    init(const std::vector<DataTensor_in*>& inputs,
         std::vector<DataTensor_out*>& outputs,
         PoolingParam<NV> &pooling_param, Context<NV> &ctx) {
            
        this->_ctx = &ctx;
        
        cudaStream_t cuda_stream;
        cuda_stream = ctx.get_compute_stream();
        
        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, cuda_stream));
        
        cudnn::createTensorDesc<float>(&_input_descs);
        cudnn::createTensorDesc<float>(&_output_descs);
        
        cudnn::create_pooling_des<float>(&_pooling_descs);
        
        return create(inputs, outputs, pooling_param, ctx);
}
  
template <>
SaberStatus VenderPooling<NV, AK_FLOAT>::\
    dispatch(const std::vector<DataTensor_in*>& inputs,
                    std::vector<DataTensor_out*>& outputs,
                    PoolingParam<NV> &param) {
            const float *in_data = (const float*)inputs[0]->data();
            float *out_data = (float*)outputs[0]->mutable_data();
            
            CUDNN_CHECK(cudnnPoolingForward(_handle, _pooling_descs,
                                            cudnn::cudnnTypeWrapper<float>::kOne(),
                                            _input_descs, in_data,
                                            cudnn::cudnnTypeWrapper<float>::kZero(),
                                            _output_descs, out_data
                                            ));
            
            return SaberSuccess;
}
DEFINE_OP_TEMPLATE(VenderPooling, PoolingParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(VenderPooling, PoolingParam, NV, AK_INT8);
} //namespace saber
} // namespace anakin
