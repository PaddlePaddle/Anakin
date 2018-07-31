#include "saber/funcs/impl/x86/saber_conv.h"
#include "saber/funcs/impl/x86/x86_utils.h"
namespace anakin {

namespace saber {

template<>
SaberStatus SaberConv2D<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
        const std::vector<DataTensor_in*>& inputs, \
        std::vector<DataTensor_out*>& outputs, ConvParam <OpTensor>& param) {
    const OpDataType* bias_ptr= nullptr;
    bool with_bias=false;
    if(param.bias()!= nullptr&&param.bias()->data()!= nullptr){
        bias_ptr=param.bias()->data();
        with_bias=true;
    }
//    conv_basic_x86(*outputs[0],*inputs[0],param.weight()->data(),bias_ptr,
//               param.group,param.weight()->width(),param.weight()->height(),
//               param.stride_w,param.stride_h,param.dilation_w,param.dilation_h,
//               param.pad_w,param.pad_h,with_bias,false);
    im2col_conv_cpu(*outputs[0],*inputs[0],_im2col_workspace,param.weight()->data(),bias_ptr,
                    param.group,param.weight()->width(),param.weight()->height(),
                    param.stride_w,param.stride_h,param.dilation_w,param.dilation_h,
                    param.pad_w,param.pad_h,with_bias,false);
};
template class SaberConv2D<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;



}
}