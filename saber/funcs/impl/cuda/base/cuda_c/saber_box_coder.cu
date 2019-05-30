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

#include "saber/funcs/impl/cuda/saber_box_coder.h"

namespace anakin {

namespace saber {

enum BOX_CODER_VAR {
    FIX_SIZE_VAR = 0,
    NO_VAR = 1,
    FROM_INPUT_VAR = 2
};

template <BOX_CODER_VAR fix_size_var, bool with_scale_clamp>
__global__ void decode_center_size_kernel(
    const float* prior_box_data, const float* prior_box_var_data,
    const float* target_box_data, const int row, const int col, const int anchor_len,
    const int axis, float* output, float nomalized, float max_scale, float min_scale) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int prior_box_offset = 0;
    int out_len = 4;
    int var_len = 4;
    int delta_len = 4;

    if (idx < row * col) {
        const int col_idx = idx % col;
        const int row_idx = idx / col;
        prior_box_offset = axis == 0 ? col_idx * anchor_len : row_idx * anchor_len;
        prior_box_offset += 1;
        float prior_box_width = prior_box_data[prior_box_offset + 2] -
                                prior_box_data[prior_box_offset] + nomalized;
        float prior_box_height = prior_box_data[prior_box_offset + 3] -
                                 prior_box_data[prior_box_offset + 1] + nomalized;
        float prior_box_center_x =
            prior_box_data[prior_box_offset] + prior_box_width * 0.5f;
        float prior_box_center_y =
            prior_box_data[prior_box_offset + 1] + prior_box_height * 0.5f;

        float box_var_x = 1.f;
        float box_var_y = 1.f;
        float box_var_w = 1.f;
        float box_var_h = 1.f;

        if (fix_size_var == FROM_INPUT_VAR) {
            int prior_var_offset = axis == 0 ? col_idx * var_len : row_idx * var_len;
            box_var_x = prior_box_var_data[prior_var_offset];
            box_var_y = prior_box_var_data[prior_var_offset + 1];
            box_var_w = prior_box_var_data[prior_var_offset + 2];
            box_var_h = prior_box_var_data[prior_var_offset + 3];
        } else if (fix_size_var == FIX_SIZE_VAR) {
            box_var_x = prior_box_var_data[0];
            box_var_y = prior_box_var_data[1];
            box_var_w = prior_box_var_data[2];
            box_var_h = prior_box_var_data[3];
        }

        float scale_width = expf(box_var_w * target_box_data[idx * delta_len + 2]);
        float scale_height = expf(box_var_h * target_box_data[idx * delta_len + 3]);

        if (with_scale_clamp) {
            scale_width = fmaxf(fminf(scale_width, max_scale), min_scale);
            scale_height = fmaxf(fminf(scale_height, max_scale), min_scale);
        }

        float target_box_width =
            scale_width * prior_box_width;
        float target_box_height =
            scale_height * prior_box_height;
        float target_box_center_x =
            box_var_x * target_box_data[idx * delta_len] * prior_box_width +
            prior_box_center_x;
        float target_box_center_y =
            box_var_y * target_box_data[idx * delta_len + 1] * prior_box_height +
            prior_box_center_y;

        output[idx * out_len] = target_box_center_x - target_box_width / 2;
        output[idx * out_len + 1] = target_box_center_y - target_box_height / 2;
        output[idx * out_len + 2] =
            target_box_center_x + target_box_width / 2 - nomalized;
        output[idx * out_len + 3] =
            target_box_center_y + target_box_height / 2 - nomalized;
    }
}

template <BOX_CODER_VAR fix_size_var, bool with_scale_clamp>
__global__ void decode_center_size_torch(
    const float* prior_box_data, const float* prior_box_var_data,
    const float* target_box_data, const int row, const int col, const int anchor_len,
    const int axis, float* output, float nomalized, float max_scale, float min_scale) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int prior_box_offset = 0;
    int out_len = 4;
    int var_len = 4;
    int delta_len = 4;

    if (idx < row * col) {
        const int col_idx = idx % col;
        const int row_idx = idx / col;
        prior_box_offset = row_idx * anchor_len;
        prior_box_offset += 1;
        float prior_box_width = prior_box_data[prior_box_offset + 2] -
                                prior_box_data[prior_box_offset] + 1.f;
        float prior_box_height = prior_box_data[prior_box_offset + 3] -
                                 prior_box_data[prior_box_offset + 1] + 1.f;
        float prior_box_center_x =
            (prior_box_data[prior_box_offset] + prior_box_data[prior_box_offset + 2]) * 0.5f;
        float prior_box_center_y =
            (prior_box_data[prior_box_offset + 1] + prior_box_data[prior_box_offset + 3]) * 0.5f;

        float box_var_x = 1.f;
        float box_var_y = 1.f;
        float box_var_w = 1.f;
        float box_var_h = 1.f;

        if (fix_size_var == FROM_INPUT_VAR) {
            int prior_var_offset = axis == 0 ? col_idx * var_len : row_idx * var_len;
            box_var_x = prior_box_var_data[prior_var_offset];
            box_var_y = prior_box_var_data[prior_var_offset + 1];
            box_var_w = prior_box_var_data[prior_var_offset + 2];
            box_var_h = prior_box_var_data[prior_var_offset + 3];
        } else if (fix_size_var == FIX_SIZE_VAR) {
            box_var_x = prior_box_var_data[0];
            box_var_y = prior_box_var_data[1];
            box_var_w = prior_box_var_data[2];
            box_var_h = prior_box_var_data[3];
        }

        float scale_width = expf(box_var_w * target_box_data[idx * delta_len + 2]);
        float scale_height = expf(box_var_h * target_box_data[idx * delta_len + 3]);

        if (with_scale_clamp) {
            scale_width = fmaxf(fminf(scale_width, max_scale), min_scale);
            scale_height = fmaxf(fminf(scale_height, max_scale), min_scale);
        }

        float target_box_width =
            scale_width * prior_box_width;
        float target_box_height =
            scale_height * prior_box_height;
        float target_box_center_x =
            box_var_x * target_box_data[idx * delta_len] * prior_box_width +
            prior_box_center_x;
        float target_box_center_y =
            box_var_y * target_box_data[idx * delta_len + 1] * prior_box_height +
            prior_box_center_y;

        output[idx * out_len] = target_box_center_x - target_box_width / 2 + 0.5f;
        output[idx * out_len + 1] = target_box_center_y - target_box_height / 2 + 0.5f;
        output[idx * out_len + 2] =
            target_box_center_x + target_box_width / 2 - 0.5f;
        output[idx * out_len + 3] =
            target_box_center_y + target_box_height / 2 - 0.5f;
    }
}

template <BOX_CODER_VAR fix_size_var>
static inline void box_coder(Tensor<NV>* proposals,
                             const Tensor<NV>* anchors,
                             const Tensor<NV>* bbox_deltas,
                             const Tensor<NV>* variances,
                             BoxCoderParam<NV>& param,
                             cudaStream_t stream
                            ) {
    constexpr size_t delta_len = 4;
    const size_t row = bbox_deltas->num();
    size_t col = bbox_deltas->channel();
    bool multiclass = bbox_deltas->width() * bbox_deltas->height() == 1;

    if (multiclass) {
        col = bbox_deltas->channel() / delta_len; //col = class number
    }

    //    const size_t anchor_nums = row * col;
    const size_t len = anchors->valid_shape()[1];
    //    CHECK_EQ(len, 5) << "anchor length is 5";
    const float* anchor_data = (const float*) anchors->data();
    const float* bbox_deltas_data = (const float*) bbox_deltas->data();
    float* proposals_data = (float*) proposals->data();
    const float* variances_data = nullptr;
    float normalized = !param.box_normalized ? 1.f : 0;

    if (variances) {
        variances_data = (const float*)variances->data();
    }

    int block = 512;
    int grid = (row * col + block - 1) / block;

    if (param.min_hw_scale > 0.f) {
        //torch mode
        decode_center_size_torch<fix_size_var, true> << < grid, block, 0, stream >> > (anchor_data,
                variances_data,
                bbox_deltas_data,
                row, col, len, param.axis, proposals_data, normalized, 1.f / param.min_hw_scale,
                param.min_hw_scale);
    } else {
        decode_center_size_kernel<fix_size_var, false> << < grid, block, 0, stream >> > (anchor_data,
                variances_data,
                bbox_deltas_data,
                row, col, len, param.axis, proposals_data, normalized, 0, 0);
    }
};

template <DataType OpDtype>
SaberStatus SaberBoxCoder<NV, OpDtype>::dispatch(const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs, BoxCoderParam<NV>& param) {
    Tensor<NV>* anchor = inputs[0];
    Tensor<NV>* delta = inputs[1];
    Tensor<NV>* variances = nullptr;
    Tensor<NV>* proposal = outputs[0];

    if (param.variance() != nullptr && param.variance()->valid_size() > 0) {
        variances = param.variance();
        CHECK(variances->valid_size() == 4);
        box_coder<FIX_SIZE_VAR>(proposal, anchor, delta, variances, param,
                                this->_ctx ->get_compute_stream());
    } else if (inputs.size() >= 3) {
        variances = inputs[2];
        box_coder<FROM_INPUT_VAR>(proposal, anchor, delta, variances, param,
                                  this->_ctx ->get_compute_stream());
    } else {
        box_coder<NO_VAR>(proposal, anchor, delta, variances, param, this->_ctx ->get_compute_stream());
    }

    return SaberSuccess;
}

template class SaberBoxCoder<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberBoxCoder, BoxCoderParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberBoxCoder, BoxCoderParam, NV, AK_INT8);
} //namespace anakin

} //name
