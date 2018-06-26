#ifndef ANAKIN_SABER_FUNCS_IMPL_BMDNN_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_BMDNN_POOLING_H

#include "saber/funcs/impl/impl_pooling.h"

namespace anakin{

namespace saber {

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderPooling<BM, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>:\
 public ImplBase<
    Tensor<BM, inDtype, LayOutType_in>, 
    Tensor<BM, outDtype, LayOutType_out>,
    Tensor<BM, OpDtype, LayOutType_op>,
    PoolingParam<Tensor<BM, OpDtype, LayOutType_op>>> {
public:
    typedef Tensor<BM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<BM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<BM, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderPooling() : _handle(NULL), _pooling_type(NULL) {}

    ~VenderPooling() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  PoolingParam<OpTensor> &pooling_param, Context<BM> &ctx) {
        return create(inputs, outputs, pooling_param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
                PoolingParam<OpTensor> &pooling_param, Context<BM> &ctx) {
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          PoolingParam<OpTensor> &param) {
        const InDataType in_data = *(inputs[0]->data());
        OutDataType out_data = *(outputs[0]->mutable_data());
        int input_n = inputs[0]->num();
        int input_c = inputs[0]->channel();
        int input_h = inputs[0]->height();
        int input_w = inputs[0]->width();
        int kh = param.window_h;
        int kw = param.window_w;
        int pad_h = param.pad_h;
        int pad_w = param.pad_w;
        int stride_h = param.stride_h;
        int stride_w = param.stride_w;
        int is_avg_pooling;
        if(_pooling_type == Pooling_max){
            is_avg_pooling = 0;
        } else {
            is_avg_pooling = 1;
        }
        _handle = get_bm_handle();
        BMDNN_CHECK(bmdnn_pooling_forward(_handle, in_data, 
                            input_n, input_c, input_h, input_w, kh, kw, pad_h, pad_w, 
                            stride_h, stride_w, is_avg_pooling, out_data));
        return SaberSuccess;
    }

private:
    bm_handle_t _handle;
    PoolingType _pooling_type;
};

template class VenderPooling<BM, AK_BM, AK_BM, AK_BM, NCHW, NCHW, NCHW>;

} //namespace saber

} // namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_BMDNN_POOLING_H
