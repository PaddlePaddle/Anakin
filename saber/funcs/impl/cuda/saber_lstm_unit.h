
#ifndef SABER_FUNCS_IMPL_CUDA_LSTM_UNIT_H
#define SABER_FUNCS_IMPL_CUDA_LSTM_UNIT_H

#include "saber/core/common.h"
#include "saber/funcs/impl/impl_lstm_unit.h"
namespace anakin {

namespace saber {

template<DataType OpDtype,
         DataType inDtype,
         DataType outDtype,
         typename LayOutType_op,
         typename LayOutType_in,
         typename LayOutType_out>
class SaberLstmUnit<NV, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase <
    Tensor<NV, inDtype, LayOutType_in>,
    Tensor<NV, outDtype, LayOutType_out>,
    Tensor<NV, OpDtype, LayOutType_op>,
    LstmUnitParam<Tensor<NV, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberLstmUnit() {}

    ~SaberLstmUnit() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             LstmUnitParam<OpTensor>& param, Context<NV>& ctx) {
        this->_ctx = &ctx;
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               LstmUnitParam<OpTensor>& param, Context<NV>& ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 LstmUnitParam<OpTensor>& param);

};

}
}

#endif //SABER_FUNCS_IMPL_CUDA_LSTM_UNIT_H
