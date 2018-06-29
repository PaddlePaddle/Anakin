#ifndef ANAKIN_SABER_FUNCS_IMPL_BMDNN_BATCH_NORM_H
#define ANAKIN_SABER_FUNCS_IMPL_BMDNN_BATCH_NORM_H

#include "saber/funcs/impl/impl_batch_norm.h"

namespace anakin{

namespace saber {

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class VenderBatchNorm<BM, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>:\
 public ImplBase<
    Tensor<BM, inDtype, LayOutType_in>, 
    Tensor<BM, outDtype, LayOutType_out>,
    Tensor<BM, OpDtype, LayOutType_op>,
    BatchnormParam<Tensor<BM, OpDtype, LayOutType_op>>> {
public:
    typedef Tensor<BM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<BM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<BM, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    VenderBatchNorm() : _handle(NULL) {}

    ~VenderBatchNorm() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                  std::vector<DataTensor_out*>& outputs,
                  BatchnormParam<OpTensor> &batch_norm_param, Context<BM> &ctx) {

        _handle = get_bm_handle();
        return create(inputs, outputs, batch_norm_param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
                BatchnormParam<OpTensor> &batch_norm_param, Context<BM> &ctx) {
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          BatchnormParam<OpTensor> &param) {

        const InDataType *in_data = (const InDataType *) inputs[0]->data();
        OutDataType *out_data = (OutDataType *) outputs[0]->mutable_data();

        int input_n = inputs[0]->num();
        int input_c = inputs[0]->channel();
        int input_h = inputs[0]->height();
        int input_w = inputs[0]->width();

        float eps = param.eps;
        float scale = param.scale;
        
        bm_device_mem_t mean_ma = bm_mem_from_system(&param.mean[0]);
        bm_device_mem_t variance_ma = bm_mem_from_system(&param.variance[0]);

        bm_device_mem_t* variance_holder = new bm_device_mem_t();

        bmdnn_batchnorm_forward_inference(
                _handle,
                //input
                *in_data,
                mean_ma,
                variance_ma,
                scale,
                *variance_holder,
                eps,
                input_n,
                input_c,
                input_h,
                input_w,
                //output
                *out_data
        );

        return SaberSuccess;
    }

private:
    bm_handle_t _handle;
};

} //namespace saber

} // namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_BMDNN_BATCH_NORM_H
