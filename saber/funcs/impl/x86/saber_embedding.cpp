
#include "saber/funcs/impl/x86/saber_embedding.h"
#include "saber/funcs/impl/x86/x86_utils.h"


namespace anakin{
namespace saber {

template class SaberEmbedding<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberEmbedding<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EmbeddingParam<OpTensor> &param,
        Context<X86> &ctx)
{
    // get context
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberEmbedding<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EmbeddingParam<OpTensor>& param,
        Context<X86> &ctx)
{
    return SaberSuccess;
}


template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberEmbedding<X86, OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EmbeddingParam<OpTensor> &param)
{
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;
    CHECK_EQ(inputs.size(), (size_t)1);
    CHECK_EQ(outputs.size(), (size_t)1);
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    const int num_word = inputs[0]->valid_size();
    auto in_data =  inputs[0]->data();
    auto out_data =  outputs[0]->mutable_data();
    int emb_dim = param.emb_dim;
    for (int i = 0; i < num_word; i++) {
        if (in_data[i] == param.padding_idx) {
            memset(out_data + i * emb_dim, 0, sizeof(DataType_out) * emb_dim);
        } else {
            CHECK_GE(in_data[i], 0);
            CHECK_LT(in_data[i], param.word_num);
            memcpy(out_data + i * emb_dim, param.weight()->data(int(in_data[i]) * emb_dim), sizeof(DataType_out) * emb_dim);
        }
    }
    return SaberSuccess;
      
}

}
} // namespace anakin
