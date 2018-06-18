#ifndef ANAKIN_SABER_FUNCS_BMDNN_ACT_H
#define ANAKIN_SABER_FUNCS_BMDNN_ACT_H
#include "saber/funcs/impl/impl_activation.h"
namespace anakin {

namespace saber {

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderActivation<BM, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<BM, inDtype, LayOutType_in>, 
        Tensor<BM, outDtype, LayOutType_out>,
        Tensor<BM, OpDtype, LayOutType_op>,
        ActivationParam<Tensor<BM, OpDtype, LayOutType_op> > > 
{
public:
    typedef Tensor<BM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<BM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<BM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderActivation()
            : _handle(NULL), _active_descs(NULL), _input_descs(NULL), _output_descs(NULL) {}

    ~VenderActivation() {
        if (_input_descs) {
            BMDNN_CHECK(bm_free_device(_input_descs));
        }
        if (_output_descs) {
            BMDNN_CHECK(bm_free_device(_output_descs));
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ActivationParam<OpTensor>& param, Context<BM>& ctx) {
        // not sure
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ActivationParam<OpTensor>& param, Context<BM>& ctx) {
        // not sure
        return SaberSuccess;
    }

    //call bmdnn activation funcs here
    virtual SaberStatus dispatch(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ActivationParam<OpTensor>& param) {

        const InDataType *in_data = (const InDataType *) inputs[0]->data();
        OutDataType *out_data = (OutDataType *) outputs[0]->mutable_data();
        int input_dim = inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();
        int input_n = inputs[0]->num();

        switch (_active_type) {
            case Active_sigmoid:
                BMDNN_CHECK(bmdnn_sigmoid_forward(_handle, _input_descs, input_n, input_dim, _output_descs));
                break;
            case Active_relu:
                BMDNN_CHECK(bmdnn_relu_forward(_handle, _input_descs, input_n, input_dim, _output_descs));
                break;
            case Active_tanh:
                BMDNN_CHECK(bmdnn_tanh_forward(_handle, _input_descs, input_n, input_dim, _output_descs));
                break;
        }
        /* BMDNN_CHECK(cudnnActivationForward(_handle, _active_descs, */
        /*                                    cudnn::cudnnTypeWrapper<InDataType>::kOne(), */
        /*                                    _input_descs, in_data, */
        /*                                    cudnn::cudnnTypeWrapper<InDataType>::kZero(), */
        /*                                    _output_descs, out_data */
        /* )); */
        return SaberSuccess;
    }

private:
    bm_handle_t _handle;
    bm_device_mem_t _input_descs;
    bm_device_mem_t _output_descs;
    ActiveType _active_type;
};
template class VenderActivation<BM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}
}

#endif //ANAKIN_SABER_FUNCS_BMDNN_ACT_H
