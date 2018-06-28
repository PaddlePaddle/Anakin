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
    BatchNormParam<Tensor<BM, OpDtype, LayOutType_op>>> {
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
                  BatchNormParam<OpTensor> &batch_norm_param, Context<BM> &ctx) {

        _handle = get_bm_handle();
        return create(inputs, outputs, batch_norm_param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                std::vector<DataTensor_out*>& outputs,
                BatchNormParam<OpTensor> &batch_norm_param, Context<BM> &ctx) {
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          BatchNormParam<OpTensor> &param) {

        return SaberSuccess;
    }

private:
    bm_handle_t _handle;
};

} //namespace saber

} // namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_BMDNN_BATCH_NORM_H
