#ifndef ANAKIN_SABER_FUNCS_BMDNN_FC_H
#define ANAKIN_SABER_FUNCS_BMDNN_FC_H

#include "saber/funcs/impl/impl_fc.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class VenderFc<BM, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>: \
    public ImplBase<
        Tensor<BM, inDtype, LayOutType_in>, \
        Tensor<BM, outDtype, LayOutType_out>, \
        Tensor<BM, OpDtype, LayOutType_op>, \
        FcParam<Tensor<BM, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<BM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<BM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<BM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderFc(): _handle(NULL) {};
    ~VenderFc() {}

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            FcParam<OpTensor>& param, Context<BM>& ctx){
        _handle = get_bm_handle();
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            FcParam<OpTensor>& param, Context<BM>& ctx){
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            FcParam<OpTensor>& param){
        const InDataType *in_data = (const InDataType *) inputs[0]->data();
        const InDataType *weights = (const InDataType *) param.weights->data();
        const InDataType *bias = (const InDataType *) param.bias->data();
        OutDataType *out_data = (OutDataType *) outputs[0]->mutable_data();
        int batch_size = inputs[0]->count_valid(0, param.axis);
        int input_len = inputs[0]->count_valid(param.axis, inputs[0]->dims());
        int output_len = param.num_output;
        if (output_len <= 0) {
            int weight_size = param.weights->valid_size();
            output_len = weight_size / input_len;
        }

        BMDNN_CHECK(bmdnn_fc_forward(_handle, *in_data, *weights, *bias,
                                    batch_size, output_len, input_len, param.is_transpose_weights, 1, 0,
                                    *out_data));
        return SaberSuccess;
    };

private:
    bm_handle_t _handle;
};

} //namespace saber

} //namespace anakin

#endif // ANAKIN_SABER_FUNCS_BMDNN_FC_H
