#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/x86/saber_conv_act.h"
#include "saber/funcs/impl/x86/jit_call_conf.h"
#include "saber/funcs/impl/x86/jit_uni_dw_convolution.h"
#include "saber/funcs/impl/x86/jit_avx512_conv1x1_act.h"
#include "saber/funcs/impl/x86/jit_avx512_conv_act.h"
#include "saber/funcs/impl/x86/jit_avx2_conv_act.h"

namespace anakin {
namespace saber {

//#define INTEL_COM

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberConv2DAct<X86, OpDtype, inDtype, outDtype,
    LayOutType_op, LayOutType_in, LayOutType_out>::init(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    ConvActiveParam<OpTensor> &param,
    Context<X86> &ctx)
{

    SaberStatus ret = SaberUnImplError;

    ConvParam<OpTensor> *conv_param = &(param.conv_param);
    const OpTensor *weight = conv_param->weight();
    Shape weight_shape(weight->shape());

    // go to different engines per different input parameters
    if(std::is_same<LayOutType_out, NCHW>::value&&std::is_same<LayOutType_in, NCHW>::value&&std::is_same<LayOutType_op, NCHW>::value){
        return SaberSuccess;
    }
    else if (conv_param->group == weight_shape[0] && conv_param->group == weight_shape[1]) {
        // depth-wise convolution
        if (this->impl) {
            delete this->impl;
        }
        this->impl = new JitUniDWConvolution<OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->impl->init(inputs, outputs, param, ctx);
        if (ret == SaberSuccess) {
//            LOG(INFO) << "++++++++++++JitUniDWConvolution";
            return ret;
        }
    } else if (weight_shape[2] == 1 && weight_shape[3] == 1) {
        // 1x1 convolution+act
        if (this->impl) {
            delete this->impl;
        }
        this->impl = new JitAvx512Conv1x1Act<OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->impl->init(inputs, outputs, param, ctx);
        if (ret == SaberSuccess) {
//            LOG(INFO) << "++++++++++++JitAvx512Conv1x1Act";
            return ret;
        }
    } else if (std::is_same<LayOutType_out, NCHW_C16>::value) {
        if (this->impl) {
            delete this->impl;
        }
        this->impl = new JitAvx512ConvAct<OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->impl->init(inputs, outputs, param, ctx);
        if (ret == SaberSuccess) {
//            LOG(INFO) << "++++++++++++JitAvx512ConvAct";
            return ret;
        }
    } else if (std::is_same<LayOutType_out, NCHW_C8>::value) {
        if (this->impl) {
            delete this->impl;
        }
        this->impl = new JitAvx2ConvAct<OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->impl->init(inputs, outputs, param, ctx);
        if (ret == SaberSuccess) {
//            LOG(INFO) << "++++++++++++JitAvx2ConvAct";
            return ret;
        }
    }
    return SaberUnImplError;

}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberConv2DAct<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ConvActiveParam<OpTensor> &param,
        Context<X86> &ctx)
{

    if(std::is_same<LayOutType_out, NCHW>::value&&std::is_same<LayOutType_in, NCHW>::value&&std::is_same<LayOutType_op, NCHW>::value){
        return SaberSuccess;
    }
    SaberStatus ret = SaberSuccess;
    if (!this->impl) {
        LOG(ERROR) << "impl is NULL";
        return SaberNotInitialized;
    }
    ret = this->impl->create(inputs, outputs, param, ctx);
    return ret;

}

template<typename DataTensor_in,typename DataTensor_out>
void conv_basic(DataTensor_in* tensor_out, DataTensor_out* tensor_in,
                const float *weights, const float *bias, int group,
                int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h,
                int pad_w, int pad_h, bool flag_bias, bool flag_relu) {

    int w_in = tensor_in->width();
    int h_in = tensor_in->height();
    int ch_in = tensor_in->channel();
    int num_in = tensor_in->num();

    int w_out = tensor_out->width();
    int h_out = tensor_out->height();
    int ch_out = tensor_out->channel();

    const int size_kernel = kernel_h * kernel_w;

    int kernel_ext_w = (kernel_w - 1) * dila_w + 1;
    int kernel_ext_h = (kernel_h - 1) * dila_h + 1;

    const int ch_out_g = ch_out / group;
    const int ch_in_g = ch_in / group;
    const int size_in_channel = w_in * h_in;
    const int size_in_batch = size_in_channel * ch_in;
    const int size_out_channel = w_out * h_out;
    const int size_out_batch = size_out_channel * ch_out;
    const float *data_in = tensor_in->data();
    float *outptr = tensor_out->mutable_data();

    for (int b = 0; b < num_in; ++b) {
        float *outptr_batch = outptr + b * size_out_batch;
        const float *data_in_batch = data_in + b * size_in_batch;
        for (int g = 0; g < group; ++g) {
            for (int c = 0; c < ch_out_g; ++c) {
                const float *inptr_group = data_in_batch + g * ch_in_g * size_in_channel;
                float *outptr_ch = outptr_batch + (g * ch_out_g + c) * size_out_channel;
                const float *weight_ch = weights + (g * ch_out_g + c) * ch_in_g * size_kernel;

                float bias_value = flag_bias ? bias[g * ch_out_g + c] : 0.f;
//                fill_bias(outptr_ch, &bias_value, 1, w_out * h_out);

                for (int i = 0; i < h_out; ++i) {
                    for (int j = 0; j < w_out; ++j) {

                        const float *weight_ch_in = weight_ch;

                        int hstart = i * stride_h - pad_h;
                        int wstart = j * stride_w - pad_w;
                        int hend = std::min(hstart + kernel_ext_h, h_in);
                        int wend = std::min(wstart + kernel_ext_w, w_in);
                        hstart = std::max(hstart, 0);
                        wstart = std::max(wstart, 0);

                        int khstart = hend < kernel_ext_h ? (kernel_ext_h - hend) / dila_h : 0;
                        int kwstart = wend < kernel_ext_w ? (kernel_ext_w - wend) / dila_w : 0;

                        const float *inptr_ch = inptr_group + hstart * w_in + wstart;

                        for (int k = 0; k < ch_in_g; ++k) {
                            const float *inptr_kernel = inptr_ch;
                            int khidx = khstart;
                            for (int idxh = hstart; idxh < hend; idxh += dila_h, khidx++) {
                                const float *inptr_kernel_w = inptr_kernel;
                                int kwidx = kwstart;
                                for (int idxw = wstart; idxw < wend; idxw += dila_w, kwidx++) {
                                    outptr_ch[j] += weight_ch_in[khidx * kernel_w + kwidx] * inptr_kernel_w[0];
                                    inptr_kernel_w += dila_w;
                                }
                                inptr_kernel += dila_h * w_in;
                            }
                            inptr_ch += size_in_channel;
                            weight_ch_in += size_kernel;
                        }
                        if (flag_bias) {
                            outptr_ch[j] += bias_value;
                        }
                        if (flag_relu) {
                            outptr_ch[j] = outptr_ch[j] > 0 ? outptr_ch[j] : 0.f;
                        }
//                        LOG(INFO)<<"out = "<<outptr_ch[j];
                    }
                    outptr_ch += w_out;
                }
            }
        }
    }
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberConv2DAct<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        ConvActiveParam<OpTensor> &param)
{

    if(std::is_same<LayOutType_out, NCHW>::value&&std::is_same<LayOutType_in, NCHW>::value&&std::is_same<LayOutType_op, NCHW>::value){
        const float* bias_ptr= nullptr;
        bool with_bias=false;
        if(param.conv_param.bias()!= nullptr){
            bias_ptr=param.conv_param.bias()->data();
            with_bias=true;
        }
        bool with_relu=false;
        if(param.has_active&&param.activation_param.active==Active_relu){
            with_relu=true;
        }
        CHECK_NOTNULL(outputs[0])<<"outputs can not be null";
        conv_basic(outputs[0],inputs[0],param.conv_param.weight()->data(),bias_ptr,
                   param.conv_param.group,param.conv_param.weight()->width(),param.conv_param.weight()->height(),
                   param.conv_param.stride_w,param.conv_param.stride_h,param.conv_param.dilation_w,param.conv_param.dilation_h,
                   param.conv_param.pad_w,param.conv_param.pad_h,with_bias,with_relu);
        return SaberSuccess;
    }

    SaberStatus ret = SaberSuccess;
    if (!this->impl) {
        CHECK(false) << "impl is NULL";
        return SaberNotInitialized;
    }
    ret = this->impl->dispatch(inputs, outputs, param);
    return ret;

}



template class SaberConv2DAct<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16>;
template class SaberConv2DAct<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C16>;
template class SaberConv2DAct<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C8>;

template class SaberConv2DAct<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;


}
} // namespace anakin
