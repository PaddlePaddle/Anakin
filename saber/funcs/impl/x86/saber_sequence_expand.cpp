
#include "saber/funcs/impl/x86/saber_sequence_expand.h"
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberSequenceExpand<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SequenceExpandParam<OpTensor> &param,
        Context<X86> &ctx)
{

    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberSequenceExpand<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SequenceExpandParam<OpTensor> &param,
        Context<X86> &ctx)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    this->_ctx = &ctx;

    return SaberSuccess;
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberSequenceExpand<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SequenceExpandParam<OpTensor> &param)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    // TODO !! need add other types of sequence_expand
    auto cur_offset = inputs[0]->get_seq_offset();
    auto ref_offset = inputs[1]->get_seq_offset();
    size_t len = inputs[0]->valid_size();
    DataType_in *input_data = inputs[0]->data();
    DataType_out *output_data = outputs[0]->mutable_data();
    int dim = inputs[0]->valid_size() / inputs[0]->num();
    if (cur_offset.size() == 0) {
        for (int i = 0; i < ref_offset.size() - 1; i++) {
             for (int j = ref_offset[i]; j < ref_offset[i+1]; j++) {
                  memcpy(output_data + j * dim, input_data + i * dim, sizeof(DataType_in) * dim);
             }
        }
        outputs[0]->set_seq_offset(ref_offset);
    } else {
        std::vector<int> out_offset;
        int cum = 0;
        for (int i = 0; i < ref_offset.size() - 1; i++) {
             int cur_len = cur_offset[i + 1] - cur_offset[i];
             for (int j = ref_offset[i]; j < ref_offset[i+1]; j++) {
                 
                  memcpy(output_data + cum * dim, input_data + i * dim, sizeof(DataType_in) * dim * cur_len);
                  cum += cur_len;
                  out_offset.push_back(cum);
             }
        }
        outputs[0]->set_seq_offset(out_offset);
    }

    return SaberSuccess;
}

template class SaberSequenceExpand<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}
} // namespace anakin
