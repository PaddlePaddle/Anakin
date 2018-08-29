
#include "saber/funcs/impl/x86/saber_mvn.h"
#include "saber/funcs/impl/x86/x86_utils.h"


namespace anakin{
namespace saber {
/**
 * @brief for each graph, do MVN(Mean-Variance Normalization):
 *          formula:
 *                 (x - mean) / ( sqrt(var) + eps )  (the eps iterm avoid to divde 0).
 *        
 * 
 * @tparam OpDtype 
 * @param inputs 
 * @param outputs 
 * @param param 
 * @return SaberStatus 
 */
template <DataType OpDtype>
SaberStatus SaberMvn<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        MvnParam<X86>& param)
{
    int N = inputs[0]->num();
    int C = inputs[0]->channel();
    int H = inputs[0]->height();
    int W = inputs[0]->width();

    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();
    int num = N * C;
    int inner_dim = H * W;
    if (param.across_channels) {
        num = N;
        inner_dim *= C; //CHW
    }

    for (int i = 0; i < num; i++) {
        OpDataType mean = 0;
        OpDataType std = 0;
        OpDataType* dst_ptr = dst + i * inner_dim;
        const OpDataType* src_ptr = src + i * inner_dim;
        //compute mean
        for (int j = 0; j < inner_dim; j++) {
            mean += src_ptr[j];
        }
        mean /= inner_dim;
        //compute variance
        for (int j = 0; j < inner_dim; ++j) {
            std += (src_ptr[j] - mean) * (src_ptr[j] - mean);
        }
        std /= inner_dim;
        std = 1.0f / (sqrtf(std) + param.eps);
        // normalize: (x - mean)/(sqrt(var)+eps)
        if (param.normalize_variance) {
            for (int j = 0; j < inner_dim; j++) {
                dst_ptr[j] = (src_ptr[j] - mean) * std;
            }
        }else { // normalize: x-mean;
            for (int j = 0; j < inner_dim; j++) {
                dst_ptr[j] = src_ptr[j] - mean;
            }
        }
    }  
}

template class SaberMvn<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberMvn, MvnParam, X86, AK_INT8);
DEFINE_OP_TEMPLATE(SaberMvn, MvnParam, X86, AK_INT16);
}
} // namespace anakin