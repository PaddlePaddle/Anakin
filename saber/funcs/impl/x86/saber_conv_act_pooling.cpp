
#include "saber/funcs/impl/x86/saber_conv_act_pooling.h"
#include "saber/funcs/impl/x86/saber_conv_act.h"
#include "saber/funcs/impl/x86/saber_pooling.h"
#include "saber/funcs/impl/x86/jit_call_conf.h"

namespace anakin {
namespace saber {

using namespace jit;

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus SaberConv2DActPooling<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::init(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
                ConvActivePoolingParam<OpTensor>& param,
Context<X86>& ctx) {
    SaberStatus ret = SaberUnImplError;

    Convact_param_t c_param(param.conv_param, param.activation_param);
    Pooling_param_t p_param = param.pooling_param;

    if (!((std::is_same<LayOutType_in, NCHW>::value
            && std::is_same<LayOutType_out, NCHW_C16>::value
            && std::is_same<LayOutType_op, NCHW>::value)
            || (std::is_same<LayOutType_in, NCHW_C16>::value
                && std::is_same<LayOutType_out, NCHW_C16>::value
                && std::is_same<LayOutType_op, NCHW>::value)
            || (std::is_same<LayOutType_in, NCHW>::value
                && std::is_same<LayOutType_out, NCHW>::value
                && std::is_same<LayOutType_op, NCHW>::value)
         )) {
        return ret;
    }

    if(std::is_same<LayOutType_in, NCHW>::value
       && std::is_same<LayOutType_out, NCHW>::value
       && std::is_same<LayOutType_op, NCHW>::value){

        Shape conv_out_shape = inputs[0]->valid_shape();
        // append the $n and $c/$k, output: N * K * P * Q
        int num_idx = inputs[0]->num_index();
        int channel_idx = inputs[0]->channel_index();
        int height_idx = inputs[0]->height_index();
        int width_idx = inputs[0]->width_index();

        conv_out_shape[num_idx] = inputs[0]->num(); // N
        conv_out_shape[channel_idx] = c_param.conv_param.weight()->num(); // K

        int input_dim = inputs[0]->height(); // P
        int kernel_exten = c_param.conv_param.dilation_h * (c_param.conv_param.weight()->height() - 1) + 1;
        int output_dim = (input_dim + 2 * c_param.conv_param.pad_h - kernel_exten)
                         / c_param.conv_param.stride_h + 1;

        conv_out_shape[height_idx] = output_dim;

        input_dim = inputs[0]->width(); // Q
        kernel_exten = c_param.conv_param.dilation_w * (c_param.conv_param.weight()->width() - 1) + 1;
        output_dim = (input_dim + 2 * c_param.conv_param.pad_w - kernel_exten)
                     / c_param.conv_param.stride_w + 1;

        conv_out_shape[width_idx] = output_dim;

        _temp_tensor.reshape(conv_out_shape);
        if(buf.size()>=1){
            buf[0] = &_temp_tensor;
        }else{
            buf.push_back(&_temp_tensor);
        }




        this->c_impl = new SaberConv2DAct<X86, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->c_impl->init(inputs, buf, c_param, ctx);

        if (ret != SaberSuccess) {
            CHECK(false)<<"not success";
            return ret;
        }
        this->p_impl = new SaberPooling<X86, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->p_impl->init(buf, outputs, p_param, ctx);
        return ret;

    }else{
        Shape out = outputs[0]->shape();
        Shape shape_buf(out[0], out[1], ((out[2] - 1) * p_param.stride_h)
                                        + p_param.window_h - p_param.pad_h,
                        ((out[3] - 1) * p_param.stride_w)
                        + p_param.window_w - p_param.pad_w, 16);

        //    std::cout << "buf shape n:" << shape_buf[0]
        //              << "  c:" << shape_buf[1]
        //              << " h:" << shape_buf[2]
        //              << " w:" << shape_buf[3]
        //              << std::endl;
        DataTensor_out* b_info = new DataTensor_out(shape_buf);
        std::for_each(this->buf.begin(), this->buf.end(),
                      [&](DataTensor_out * t) {
                          delete t;
                          t = nullptr;
                      });
        buf.push_back(b_info);

        this->c_impl = new SaberConv2DAct<X86, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->c_impl->init(inputs, buf, c_param, ctx);

        if (ret != SaberSuccess) {
//            SABER_CHECK(ret);
//            CHECK(false)<<"not success";
            return ret;
        }

        this->p_impl = new SaberPooling<X86, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>;
        ret = this->p_impl->init(buf, outputs, p_param, ctx);
        return ret;
    }

}

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus SaberConv2DActPooling<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::create(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
                ConvActivePoolingParam<OpTensor>& param,
Context<X86>& ctx) {
    SaberStatus ret = SaberSuccess;

    if (!this->c_impl || !this->p_impl) {
        LOG(ERROR) << "impl is NULL";
        return SaberNotInitialized;
    }

    Convact_param_t c_param(param.conv_param, param.activation_param);
    Pooling_param_t p_param = param.pooling_param;

    Shape out = outputs[0]->shape();
    Shape shape_buf(out[0], out[1], ((out[2] - 1) * p_param.stride_h)
                    + p_param.window_h - p_param.pad_h,
                    ((out[3] - 1) * p_param.stride_w)
                    + p_param.window_w - p_param.pad_w, 16);

    LOG(INFO) << "create buf shape n:" << shape_buf[0]
              << "  c:" << shape_buf[1]
              << " h:" << shape_buf[2]
              << " w:" << shape_buf[3]
              << std::endl;

    DataTensor_out* b_info = new DataTensor_out(shape_buf);
    std::for_each(this->buf.begin(), this->buf.end(),
    [&](DataTensor_out * t) {
        delete t;
        t = nullptr;
    });
    buf.push_back(b_info);

    ret = this->c_impl->create(inputs, buf, c_param, ctx);

    if (ret != SaberSuccess) {
        return ret;
    }

    ret = this->p_impl->create(buf, outputs, p_param, ctx);
    return ret;
}

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus SaberConv2DActPooling<X86, OpDtype, inDtype, outDtype,
            LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
                const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
ConvActivePoolingParam<OpTensor>& param) {
    SaberStatus ret = SaberSuccess;

    if (!this->c_impl || !this->p_impl) {
        LOG(ERROR) << "impl is NULL";
        return SaberNotInitialized;
    }

    Convact_param_t c_param(param.conv_param, param.activation_param);
    Pooling_param_t p_param = param.pooling_param;
    ret = this->c_impl->dispatch(inputs, buf, c_param);

    if (ret != SaberSuccess) {
        return ret;
    }

    ret = this->p_impl->dispatch(buf, outputs, p_param);
    return ret;
}
template class SaberConv2DActPooling<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16>;
template class SaberConv2DActPooling<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}
} // namespace anakin
