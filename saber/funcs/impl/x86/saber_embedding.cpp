
#include "saber/funcs/impl/x86/saber_embedding.h"
#include "saber/funcs/impl/x86/x86_utils.h"


namespace anakin{
namespace saber {


template <DataType OpDtype>
SaberStatus SaberEmbedding<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        EmbeddingParam<X86> &param,
        Context<X86> &ctx)
{
    // get context
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberEmbedding<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs, 
        std::vector<Tensor<X86>*>& outputs, 
        EmbeddingParam<X86> &param, 
        Context<X86> &ctx)
{
    return SaberSuccess;
}


template <DataType OpDtype>
SaberStatus SaberEmbedding<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs, 
        std::vector<Tensor<X86>*>& outputs, 
        EmbeddingParam<X86> &param)
{

    typedef typename DataTrait<X86, OpDtype>::Dtype DataType_out;
    CHECK_EQ(inputs.size(), (size_t)1);
    CHECK_EQ(outputs.size(), (size_t)1);
    CHECK_EQ(inputs[0]->get_dtype(), AK_FLOAT) << "embedding only support float inputs!";
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    const int num_word = inputs[0]->valid_size();
    
    //outputs: chose corresponding informations of words.
    //inputs: word_id [Its type maybe float or int]
    //outputs = weights[inputs[j]].
 
    const float *in_data =  (const float*)inputs[0]->data();
    DataType_out *out_data =  (DataType_out*)outputs[0]->mutable_data();
    int emb_dim = param.emb_dim;
    for (int i = 0; i < num_word; i++) {
        if (in_data[i] == param.padding_idx) {
            memset(out_data + i * emb_dim, 0, sizeof(DataType_out) * emb_dim);
            } else {
            CHECK_GE(in_data[i], 0);
            CHECK_LT(in_data[i], param.word_num);
            memcpy(out_data + i * emb_dim, (DataType_out*)param.weight()->data()+int(in_data[i]) * emb_dim, sizeof(DataType_out) * emb_dim);
        }
    }
    
}

template class SaberEmbedding<X86, AK_FLOAT>;
template class SaberEmbedding<X86, AK_INT8>;
DEFINE_OP_TEMPLATE(SaberEmbedding, EmbeddingParam, X86, AK_INT16);
}
} // namespace anakin