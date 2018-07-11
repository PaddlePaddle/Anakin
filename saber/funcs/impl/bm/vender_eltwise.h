#ifndef ANAKIN_SABER_FUNCS_IMPL_BMDNN_ELTWISE_H
#define ANAKIN_SABER_FUNCS_IMPL_BMDNN_ELTWISE_H

#include "saber/funcs/impl/impl_eltwise.h"

namespace anakin {

namespace saber {

template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
class VenderEltwise<BM, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>:\
public ImplBase<
        Tensor<BM, inDtype, LayOutType_in>,
        Tensor<BM, outDtype, LayOutType_out>,
        Tensor<BM, OpDtype, LayOutType_op>,
        EltwiseParam<Tensor<BM, OpDtype, LayOutType_op>>> {
public:
    typedef Tensor<BM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<BM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<BM, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderEltwise() {}

    ~VenderEltwise() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                         std::vector<DataTensor_out*>& outputs,
                         EltwiseParam<OpTensor> &param,
                         Context<BM> &ctx) {
        _handle = get_bm_handle();
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                           std::vector<DataTensor_out*>& outputs,
                           EltwiseParam<OpTensor> &param,
                           Context<BM> &ctx) {
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             EltwiseParam<OpTensor> &param) {

        int op_ = 0;
        switch (param.operation) {
            case Eltwise_prod:
                op_ = 0;
                break;
            case Eltwise_sum:
                op_ = 1;
                break;
            case Eltwise_max:
                op_ = 2;
                break;
            default:
                return SaberUnImplError;
        }

        //int input_size = inputs.size();
        //CHECK_GE(input_size, 2) << "Input size should >= 2!";

        OutDataType out_data = *(outputs[0]->mutable_data());
        int input_n = inputs[0]->num();
        int input_c = inputs[0]->channel();
        int input_h = inputs[0]->height();
        int input_w = inputs[0]->width();

        std::vector<float> coeff_ = param.coeff;
        if (coeff_.size() != inputs.size()) {
            int diff = inputs.size() - coeff_.size();
            for (int j=0; j<diff; j++) {
                coeff_.push_back(1);
            }
        }

        bm_device_mem_t* mask_data = new bm_device_mem_t();

        int flag_first = 1;
        for (int i=0; i<inputs.size(); i++){
            const InDataType in_data = *(inputs[i]->data());
            bmdnn_eltwise_forward(
                    _handle,
                    op_,
                    flag_first,
                    coeff_[i],
                    i,
                    in_data,
                    out_data,
                    input_n,
                    input_c * input_h * input_w,
                    *mask_data,
                    out_data);

            bm_flush(_handle);
            flag_first = 0;
        }

        //bm_free_device(_handle, *mask_data);
        return SaberSuccess;
    }

private:
    bm_handle_t _handle;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_BMDNN_ELTWISE_H