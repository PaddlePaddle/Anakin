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

#include "saber/funcs/impl/x86/saber_box_clip.h"


namespace anakin {

namespace saber {

template<DataType OpDtype>
SaberStatus SaberBoxClip<X86, OpDtype>::dispatch(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        BoxClipParam<X86>& param) {

    static constexpr int im_info_size = 3;
    static constexpr int box_info_size = 4;
    auto seq_offset = inputs[1]->get_seq_offset();
    CHECK_EQ(inputs.size(), 2) << "need two input";
    CHECK_EQ(seq_offset.size(), 1) << "need offset to cal batch";
    CHECK_GT(seq_offset[0].size(), 1) << "need offset to cal batch";
    auto offset = seq_offset[0];
    auto img = inputs[1];
    auto im_info = inputs[0];
    const float* im_info_ptr = static_cast<const float*>(im_info->data());
    const float* box_ptr_in = static_cast<const float*>(img->data());
    float* box_ptr_out = static_cast<float*>(outputs[0]->data());
    int batch_size = offset.size() - 1;
    CHECK_EQ(batch_size * im_info_size, im_info->valid_size()) << "im_info should be valid";

    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        const float img_h = im_info_ptr[batch_id * im_info_size + 0];
        const float img_w = im_info_ptr[batch_id * im_info_size + 1];
        const float scale = im_info_ptr[batch_id * im_info_size + 2];

        if (param.is_ori_box) {
            const float img_h_scale = round(img_h / scale) - 1;
            const float img_w_scale = round(img_w / scale) - 1;
            const int start_in_batch = offset[batch_id];
            const int end_in_batch = offset[batch_id + 1];

            for (int im_id = start_in_batch; im_id < end_in_batch; im_id++) {
                const float* batch_box_ptr_in = &box_ptr_in[im_id * box_info_size];
                float* batch_box_ptr_out = &box_ptr_out[im_id * box_info_size];
                batch_box_ptr_out[0] = std::max(std::min(batch_box_ptr_in[0], img_w_scale), 0.f);
                batch_box_ptr_out[1] = std::max(std::min(batch_box_ptr_in[1], img_h_scale), 0.f);
                batch_box_ptr_out[2] = std::max(std::min(batch_box_ptr_in[2], img_w_scale), 0.f);
                batch_box_ptr_out[3] = std::max(std::min(batch_box_ptr_in[3], img_h_scale), 0.f);
            }
        } else {
            const float scale_rev = 1.f / scale;
            const float img_h_scale = img_h - 1;
            const float img_w_scale = img_w - 1;
            const int start_in_batch = offset[batch_id];
            const int end_in_batch = offset[batch_id + 1];

            for (int im_id = start_in_batch; im_id < end_in_batch; im_id++) {
                const float* batch_box_ptr_in = &box_ptr_in[im_id * box_info_size];
                float* batch_box_ptr_out = &box_ptr_out[im_id * box_info_size];
                batch_box_ptr_out[0] = std::max(std::min(batch_box_ptr_in[0], img_w_scale), 0.f) * scale_rev;
                batch_box_ptr_out[1] = std::max(std::min(batch_box_ptr_in[1], img_h_scale), 0.f) * scale_rev;
                batch_box_ptr_out[2] = std::max(std::min(batch_box_ptr_in[2], img_w_scale), 0.f) * scale_rev;
                batch_box_ptr_out[3] = std::max(std::min(batch_box_ptr_in[3], img_h_scale), 0.f) * scale_rev;
            }
        }
    }

    return SaberSuccess;
}

template
class SaberBoxClip<X86, AK_FLOAT>;

DEFINE_OP_TEMPLATE(SaberBoxClip, BoxClipParam, X86, AK_HALF);

DEFINE_OP_TEMPLATE(SaberBoxClip, BoxClipParam, X86, AK_INT8);
} //namespace anakin

} //namespace anakin
