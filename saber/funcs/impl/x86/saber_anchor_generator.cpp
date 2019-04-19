#include "saber/funcs/impl/x86/saber_anchor_generator.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include <cmath>
namespace anakin {
namespace saber {


/**
 *  @brief  formula: (k + alpha * sigma((x(i))^2)) ^ beta.
 *             where,
 *                   local_size = 5(default), means 5 channels in succession.
 *                   sigma((x(i))^2): sum of x^2 of k channels in succession.
 * 
 * 
 */
template <DataType OpDtype>
SaberStatus SaberAnchorGenerator<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    AnchorGeneratorParam<X86>& param) {
    
    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* var = (OpDataType*)outputs[1]->mutable_data();
    auto anchor_sizes = param.anchor_sizes;
    auto aspect_ratios = param.aspect_ratios;
    auto stride = param.stride;
    auto variances = param.variances;
    auto offset = param.offset;
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int stride_w = stride[0];
    int stride_h = stride[1];
    auto anchor_tmp = dst;
    auto var_tmp = var;
    for (int h_idx = 0; h_idx < height; h_idx++) {
        for (int w_idx = 0; w_idx < width; w_idx++) {
            OpDataType x_ctr = (w_idx * stride_w) + offset * (stride_w - 1);
            OpDataType y_ctr = (h_idx * stride_h) + offset * (stride_h - 1);
            for (size_t r = 0; r < aspect_ratios.size(); r++) {
                auto ar = aspect_ratios[r];
                for (size_t s = 0; s < anchor_sizes.size(); s++) {
                    auto anchor_size = anchor_sizes[s];
                    OpDataType area = stride_w * stride_h;
                    OpDataType area_ratios = area / ar;
                    OpDataType base_w = round(sqrt(area_ratios));
                    OpDataType base_h = round(base_w * ar);
                    OpDataType scale_w = anchor_size / stride_w;
                    OpDataType scale_h = anchor_size / stride_h;
                    OpDataType half_width = 0.5 * (scale_w * base_w - 1);
                    OpDataType half_height = 0.5 * (scale_h * base_h - 1);
                    anchor_tmp[0] = x_ctr - half_width;
                    anchor_tmp[1] = y_ctr - half_height;
                    anchor_tmp[2] = x_ctr + half_width;
                    anchor_tmp[3] = y_ctr + half_height;
                    var_tmp[0] = variances[0];
                    var_tmp[1] = variances[1];
                    var_tmp[2] = variances[2];
                    var_tmp[3] = variances[3];
                    anchor_tmp += 4;
                    var_tmp += 4;
                }
            }
        }
    }
    
    return SaberSuccess;
}

template class SaberAnchorGenerator<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberAnchorGenerator, AnchorGeneratorParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberAnchorGenerator, AnchorGeneratorParam, X86, AK_INT8);
}
}
