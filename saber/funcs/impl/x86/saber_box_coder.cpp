/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "saber/funcs/impl/x86/saber_box_coder.h"

namespace anakin {

namespace saber {

enum BOX_CODER_VAR {
    FIX_SIZE_VAR = 0,
    NO_VAR = 1,
    FROM_INPUT_VAR = 2
};

/**
 * NOTE: Fluid box coder no exp clamp
 * @tparam Dtype
 * @tparam fix_size_var
 * @param proposals
 * @param anchors
 * @param bbox_deltas
 * @param variances
 * @param param
 */
template <typename Dtype, BOX_CODER_VAR fix_size_var>
static inline void box_coder(Tensor<X86>* proposals,
                             const Tensor<X86>* anchors,
                             const Tensor<X86>* bbox_deltas,
                             const Tensor<X86>* variances,
                             BoxCoderParam<X86>& param
                            ) {
    const size_t row = bbox_deltas->num();
    const size_t col = bbox_deltas->channel();
    const size_t anchor_nums = row * col;
    const size_t len = anchors->valid_shape()[1];
    CHECK_EQ(len, 5) << "anchor length is 5";
    int out_len = 4;
    int var_len = 4;
    int delta_len = 4;
    
    const Dtype* anchor_data = (const Dtype*) anchors->data();
    const Dtype* bbox_deltas_data = (const Dtype*) bbox_deltas->data();
    Dtype* proposals_data = (Dtype*) proposals->data();
    const Dtype* variances_data = nullptr;
    float normalized = !param.box_normalized ? 1.f : 0;

    if (variances) {
        variances_data = (const Dtype*)variances->data();
    }

    for (int64_t row_id = 0; row_id < row; ++row_id) {
        for (int64_t col_id = 0; col_id < col; ++col_id) {
            size_t delta_offset = row_id * col * delta_len + col_id * delta_len;
            int prior_box_offset = param.axis == 0 ? col_id * len : row_id * len;
            auto anchor_data_tmp = anchor_data + prior_box_offset + 1;
            auto bbox_deltas_data_tmp = bbox_deltas_data + delta_offset;
            auto proposals_data_tmp = proposals_data + delta_offset;
            auto anchor_width = anchor_data_tmp[2] - anchor_data_tmp[0] + normalized;
            auto anchor_height = anchor_data_tmp[3] - anchor_data_tmp[1] + normalized;
            auto anchor_center_x = anchor_data_tmp[0] + 0.5 * anchor_width;
            auto anchor_center_y = anchor_data_tmp[1] + 0.5 * anchor_height;
            Dtype bbox_center_x = 0, bbox_center_y = 0;
            Dtype bbox_width = 0, bbox_height = 0;

            if (fix_size_var == FROM_INPUT_VAR) {
                int var_offset = param.axis == 0 ? col_id * var_len : row_id * var_len;
                auto variances_data_tmp = variances_data + var_offset;
                bbox_center_x =
                    variances_data_tmp[0] * bbox_deltas_data_tmp[0] * anchor_width +
                    anchor_center_x;
                bbox_center_y = variances_data_tmp[1] *
                                bbox_deltas_data_tmp[1] * anchor_height + anchor_center_y;
                bbox_width = std::exp(variances_data_tmp[2] *
                                      bbox_deltas_data_tmp[2]) * anchor_width;
                bbox_height = std::exp(variances_data_tmp[3] *
                                       bbox_deltas_data_tmp[3]) * anchor_height;
            }

            if (fix_size_var == FIX_SIZE_VAR) {
                bbox_center_x =
                    variances_data[0] * bbox_deltas_data_tmp[0] * anchor_width +
                    anchor_center_x;
                bbox_center_y = variances_data[1] *
                                bbox_deltas_data_tmp[1] * anchor_height + anchor_center_y;
                bbox_width = std::exp(variances_data[2] *
                                      bbox_deltas_data_tmp[2]) * anchor_width;
                bbox_height = std::exp(variances_data[3] *
                                       bbox_deltas_data_tmp[3]) * anchor_height;

            } else if (fix_size_var == NO_VAR) {
                bbox_center_x =
                    bbox_deltas_data_tmp[0] * anchor_width + anchor_center_x;
                bbox_center_y =
                    bbox_deltas_data_tmp[1] * anchor_height + anchor_center_y;
                bbox_width = std::exp(bbox_deltas_data_tmp[2]) * anchor_width;
                bbox_height = std::exp(bbox_deltas_data_tmp[3]) * anchor_height;
            }

            proposals_data_tmp[0] = bbox_center_x - bbox_width / 2;
            proposals_data_tmp[1] = bbox_center_y - bbox_height / 2;
            proposals_data_tmp[2] = bbox_center_x + bbox_width / 2 - normalized;
            proposals_data_tmp[3] = bbox_center_y + bbox_height / 2 - normalized;
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberBoxCoder<X86, OpDtype>::dispatch(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs, BoxCoderParam<X86>& param) {
    Tensor<X86>* anchor = inputs[0];
    Tensor<X86>* delta = inputs[1];
    Tensor<X86>* variances = nullptr;
    Tensor<X86>* proposal = outputs[0];

    if (param.variance() != nullptr && param.variance()->valid_size() > 0) {
        variances = param.variance();
        CHECK(variances->valid_size() == 4);
        box_coder<OpDataType, FIX_SIZE_VAR>(proposal, anchor, delta, variances, param);
    } else if (inputs.size() >= 3) {
        variances = inputs[2];
        box_coder<OpDataType, FROM_INPUT_VAR>(proposal, anchor, delta, variances, param);
    } else {
        box_coder<OpDataType, NO_VAR>(proposal, anchor, delta, variances, param);
    }

    return SaberSuccess;
}

template class SaberBoxCoder<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberBoxCoder, BoxCoderParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberBoxCoder, BoxCoderParam, X86, AK_INT8);
} //namespace anakin

} //name
