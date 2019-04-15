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

#include "saber/funcs/impl/x86/mkl_packed_int8_gemm.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {

SaberStatus PackedMKLInt8Gemm::init(const bool trans_a, const bool trans_b,
                                    const int m, const int n, const int k, Tensor<X86>& b, float scale_a) {
    _scale.clear();
    if (b.get_dtype() == AK_FLOAT) {
        _int8_weights_wx.re_alloc(Shape({1, 1, k, n}), AK_INT8);
        utils::ScaleUtils::scale_gemm_xw_weights_to_nchw_host(_int8_weights_wx, b, !trans_b);
        _wx_gemm.init(trans_a, trans_b, m, n, k, 0, (int8_t*)_int8_weights_wx.data(), PACKED_MKLGEMM);
    } else if (b.get_dtype() == AK_INT8){
        _int8_weights_wx.set_scale(b.get_scale());
        _wx_gemm.init(trans_a, trans_b, m, n, k, 0, (int8_t*)b.data(), PACKED_MKLGEMM);
    } else{
        LOG(FATAL)<<"not support";
    }
    for (auto i:_int8_weights_wx.get_scale()){
        _scale.push_back(i * scale_a);
    }


    _scale_in.re_alloc(Shape({1, 1, m, k}, Layout_NCHW), AK_INT8);
    _m = m;
    _n = n;
    _k = k;
    return SaberSuccess;
}
SaberStatus PackedMKLInt8Gemm::dispatch(const float alpha, const float beta, int m,
                                        const Tensor<X86>& a, Tensor<X86>& c, Tensor<X86>* bias) {
    if (a.get_dtype() == AK_FLOAT && c.get_dtype() == AK_INT32) {
        CHECK(bias == nullptr || bias->valid_size() == 0);
        CHECK_EQ(a.get_layout(), Layout_NCHW);
        utils::try_expand_tensor(_scale_in, m * _k);
        utils::ScaleUtils::scale_fp32_int8(_scale_in, a);
        _wx_gemm.dispatch(alpha, beta, m, (int8_t*)_scale_in.data(), nullptr, (int32_t*)c.data());
    } else if (a.get_dtype() == AK_FLOAT && c.get_dtype() == AK_FLOAT) {
        CHECK_EQ(a.get_layout(), Layout_NCHW);
        utils::try_expand_tensor(_scale_in, m * _k);
        utils::ScaleUtils::scale_fp32_int8(_scale_in, a);
        _wx_gemm.dispatch(alpha, beta, m, (int8_t*)_scale_in.data(), nullptr, (int32_t*)c.data());
        CHECK(_int8_weights_wx.get_scale().size() > 0);
        float* out_fp32 = static_cast<float*>(c.mutable_data());
        const float* scale_vec = _scale.data();
        int32_t* in_epi32 =  static_cast<int32_t*>(c.data());
        if (bias == nullptr || bias->valid_size() == 0) {
            if (_scale.size() == _n) {
                for (int i = 0; i < m * _n; i++) {
                    out_fp32[i] = (float) in_epi32[i] * scale_vec[i % _n];
                }
            } else if (_scale.size() == 1) {
                float scale = scale_vec[0];

                for (int i = 0; i < m * _n; i++) {
                    out_fp32[i] = (float) in_epi32[i] * scale;
                }
            }
        } else {
            CHECK_EQ(bias->get_dtype(), AK_FLOAT);
            const float* bias_ptr = static_cast<const float*>(bias->data());
            if (_scale.size() == _n) {
                for (int i = 0; i < m * _n; i++) {
                    out_fp32[i] = (float) in_epi32[i] * scale_vec[i % _n] + bias_ptr[i % _n];
                }
            } else if (_scale.size() == 1) {
                float scale = scale_vec[0];

                for (int i = 0; i < m * _n; i++) {
                    out_fp32[i] = (float) in_epi32[i] * scale + bias_ptr[i % _n];
                }
            }
        }
    } else if (a.get_dtype() == AK_INT8 && c.get_dtype() == AK_INT32) {
        CHECK(bias == nullptr || bias->valid_size() == 0);
        _wx_gemm.dispatch(alpha, beta, m, (int8_t*)a.data(), nullptr, (int32_t*)c.data());
    } else{
        LOG(FATAL)<<"not support ";
    }
    return SaberSuccess;
}

}
}