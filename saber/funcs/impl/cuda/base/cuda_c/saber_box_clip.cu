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

#include "saber/funcs/impl/cuda/saber_box_clip.h"
#include "saber/funcs/saber_util.h"
#include "tensor_op.h"
#include "debug.h"
namespace anakin {

namespace saber {

static constexpr int ImInfoSize = 3;

template <typename Dtype, int BlockSize>
static __global__ void GPUBoxClip(const Dtype* input, const int* lod,
                                  const int width, const Dtype* im_info,
                                  Dtype* output) {
    Dtype im_w = round(im_info[blockIdx.x * ImInfoSize + 1] /
                       im_info[blockIdx.x * ImInfoSize + 2]);
    Dtype im_h = round(im_info[blockIdx.x * ImInfoSize] /
                       im_info[blockIdx.x * ImInfoSize + 2]);

    for (int i = threadIdx.x; i < (lod[blockIdx.x + 1] - lod[blockIdx.x]) * width;
            i += BlockSize) {
        int idx = lod[blockIdx.x] * width + i;
        Dtype im_size = (idx % 2 == 0) ? im_w : im_h;
        output[idx] = max(min(input[idx], im_size - 1), Dtype(0.));
    }
}

template <typename Dtype, int BlockSize>
static __global__ void GPUBoxClip_no_ori(const Dtype* input, const int* lod,
                                  const int width, const Dtype* im_info,
                                  Dtype* output) {
    Dtype im_w = round(im_info[blockIdx.x * ImInfoSize + 1]);
    Dtype im_h = round(im_info[blockIdx.x * ImInfoSize]);
    Dtype scale_rev = Dtype(1)/im_info[blockIdx.x * ImInfoSize+2];

    for (int i = threadIdx.x; i < (lod[blockIdx.x + 1] - lod[blockIdx.x]) * width;
         i += BlockSize) {
        int idx = lod[blockIdx.x] * width + i;
        Dtype im_size = (idx % 2 == 0) ? im_w : im_h;
        output[idx] = max(min(input[idx], im_size - 1), Dtype(0.))*scale_rev;
    }
}

template <DataType OpDtype>
SaberStatus SaberBoxClip<NV, OpDtype>::dispatch(const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs, BoxClipParam<NV>& param) {
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
    float* box_ptr = static_cast<float*>(img->data());
    int batch_size = offset.size() - 1;
    CHECK_EQ(batch_size * im_info_size, im_info->valid_size()) << "im_info should be valid";
    utils::try_expand_tensor(cuda_seq_offset, offset.size());
    CUDA_CHECK(cudaMemcpyAsync(cuda_seq_offset.data(), offset.data(), sizeof(int)*offset.size(),
                               cudaMemcpyHostToDevice, this->_ctx->get_compute_stream()));
    if (param.is_ori_box) {
        GPUBoxClip<float, 256> << < batch_size, 256, 0, this->_ctx->get_compute_stream() >> > (
                static_cast<float *>(img->data()), static_cast<int *>(cuda_seq_offset.data()),
                        box_info_size, static_cast<float *>(im_info->data()), static_cast<float *>(outputs[0]->data()));
    }else{
        GPUBoxClip_no_ori<float, 256> << < batch_size, 256, 0, this->_ctx->get_compute_stream() >> > (
                static_cast<float *>(img->data()), static_cast<int *>(cuda_seq_offset.data()),
                        box_info_size, static_cast<float *>(im_info->data()), static_cast<float *>(outputs[0]->data()));
    }
    return SaberSuccess;
}

template class SaberBoxClip<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberBoxClip, BoxClipParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberBoxClip, BoxClipParam, NV, AK_INT8);
} //namespace anakin

} //namespace anakin
