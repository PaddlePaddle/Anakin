#ifndef ANAKIN_SABER_FUNCS_IMPL_BMDNN_CONV2D_H
#define ANAKIN_SABER_FUNCS_IMPL_BMDNN_CONV2D_H

#include "saber/funcs/impl/impl_conv.h"
#include "saber/funcs/impl/bm/bmdnn_api.h"   

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderConv2D<BM, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<BM, inDtype, LayOutType_in>,
        Tensor<BM, outDtype, LayOutType_out>,
        Tensor<BM, OpDtype, LayOutType_op>,
        ConvParam<Tensor<BM, OpDtype, LayOutType_op> > >
{
public:
    typedef Tensor<BM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<BM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<BM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderConv2D(): _handle(NULL) {}
    ~VenderConv2D() {}

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvParam<OpTensor>& param, Context<BM>& ctx) {

        _handle = get_bm_handle();
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvParam<OpTensor>& param, Context<BM>& ctx);

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          ConvParam<OpTensor>& param) {
        const InDataType *in_data = (const InDataType *) inputs[0]->data();
        const InDataType *weight = (const InDataType *) param.weight()->data();
        const InDataType *bias = (const InDataType *) param.bias()->data();
        OutDataType *out_data = (OutDataType *) outputs[0]->mutable_data();

        int input_n = inputs[0]->num();
        int input_c = inputs[0]->channel();
        int input_h = inputs[0]->height();
        int input_w = inputs[0]->width();

        int output_n = outputs[0]->num();
        int output_c = outputs[0]->channel();
        int output_h = outputs[0]->height();
        int output_w = outputs[0]->width();

        int group = param.group;
        int kh = param.weight()->height();
        int kw = param.weight()->width();
        int pad_h = param.pad_h;
        int pad_w = param.pad_w;
        int stride_h = param.stride_h;
        int stride_w = param.stride_w;
        int dilation_h = param.dilation_h;
        int dilation_w = param.dilation_w;

        bm_tensor_4d_t input_shape = {
            input_n,
            input_c,
            input_h,
            input_w
        };

        bm_tensor_4d_t output_shape = {
            output_n,
            output_c,
            output_h,
            output_w
        };

        bm_kernel_param_t kernel_param = {
            group,
            output_c,
            input_c,
            kh,
            kw
        };

        bm_conv_param_t conv_param = {
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            0
        };

        BMDNN_CHECK(bmdnn_conv_forward(_handle, *in_data, *weight, *bias, input_shape, 
                                    kernel_param, output_shape, conv_param, 1, *out_data));
                                    
        return SaberSuccess;
    }

private:
    bm_handle_t _handle;
};

}
}
#endif //ANAKIN_SABER_FUNCS_BMDNN_CONV2D_H
