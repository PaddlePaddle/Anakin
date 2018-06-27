#ifndef ANAKIN_SABER_FUNCS_IMPL_BMDNN_SOFTMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_BMDNN_SOFTMAX_H

#include "saber/funcs/impl/impl_softmax.h"
#include "saber/saber_funcs_param.h"
#include "saber/saber_types.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderSoftmax<BM, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<BM, inDtype, LayOutType_in>,
        Tensor<BM, outDtype, LayOutType_out>,
        Tensor<BM, OpDtype, LayOutType_op>,
        SoftmaxParam<Tensor<BM, OpDtype, LayOutType_op> > >
{
public:
    typedef Tensor<BM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<BM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<BM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderSoftmax(): _handle(NULL) {}
    ~VenderSoftmax() {}

    /**
     * \brief initial all bmdnn resources here
     * @param inputs
     * @param outputs
     * @param param
     * @param ctx
     */
    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            SoftmaxParam<OpTensor>& param, Context<BM>& ctx) {

        _handle = get_bm_handle();
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            SoftmaxParam<OpTensor>& param, Context<BM> &ctx) {

    }

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          SoftmaxParam<OpTensor> &param){

        const InDataType *in_data = (const InDataType *) inputs[0]->data();
        OutDataType *out_data = (OutDataType *) outputs[0]->mutable_data();

        int input_n = inputs[0]->num();
        int input_c = inputs[0]->channel();
        int input_h = inputs[0]->height();
        int input_w = inputs[0]->width();

        /*
        int outer_num = inputs[0]->count(0, param.axis);
        int inner_num = inputs[0]->count(param.axis + 1, inputs[0]->dims());

        int N = outer_num;
        int K = inputs[0]->valid_shape()[param.axis];
        int H = inner_num;
        int W = 1;

        const int stride_w = 1;
        const int stride_h = W * stride_w;
        const int stride_c = H * stride_h;
        const int stride_n = K * stride_c;
         */

        bmdnn_softmax_forward(
                _handle,
                *in_data,
                input_n,
                input_c,
                input_h * input_w,
                *out_data
        );

        return SaberSuccess;
    }

private:
    bm_handle_t _handle;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_BMDNN_SOFTMAX_H
