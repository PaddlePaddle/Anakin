/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_FRAMEWORK_LITE_UTILS_H
#define ANAKIN_FRAMEWORK_LITE_UTILS_H

#include <string>
#include <unordered_map>

#include "framework/core/parameter.h"

namespace anakin {

namespace lite {

/**
 * \brief  update conv weights with batchnorm and scale parameters.
 */
template<typename T>
void update_weights(PBlock<T> weights, PBlock<T> bias,
					int n, int c, int h, int w, bool conv_bias_term, 
					float batchnorm_scale, float batchnorm_eps, 
					std::vector<float> batchnorm_mean, 
					std::vector<float> batchnorm_variance, 
					std::vector<float> scale_w, 
					std::vector<float> scale_b, 
					bool scale_bias_term) {
	float* weights_p = (float*)weights.h_tensor().mutable_data();
	size_t type_size = weights.h_tensor().get_dtype_size();
	if (!conv_bias_term) {
		bias.re_alloc(Shape({1,batchnorm_mean.size(),1,1}));
		void* new_bias_data = bias.h_tensor().mutable_data();
		memset(new_bias_data, 0, type_size * bias.h_tensor().size());
	}
	float* bias_p = (float*)bias.h_tensor().mutable_data();

	batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
	int chw = c*h*w;
	for (int i=0; i <n; i++ ) {
		float alpha = 1.f;
		float beta = 0.f;
		// insert batchnorm parameters
		alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
		alpha = 1.f / sqrtf(alpha);
		beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
		beta = beta * alpha;

		// insert scale parameters
		alpha = scale_w[i] * alpha;
		if (scale_bias_term) {
			beta = beta * scale_w[i] + scale_b[i];
		} else {
			beta = beta * scale_w[i];
		}
		for (int j=0; j < chw; j++) {
			weights_p[i * chw + j] *= alpha;
		}
		bias_p[i] *= alpha;
		bias_p[i] += beta;
	}
}

/**
 * \brief  update conv weights with batchnorm.
 */
template<typename T>
void update_weights(PBlock<T> weights, PBlock<T> bias,
                    int n, int c, int h, int w, bool conv_bias_term,
                    float batchnorm_scale, float batchnorm_eps,
                    std::vector<float> batchnorm_mean,
                    std::vector<float> batchnorm_variance) {
    float* weights_p = (float*)weights.h_tensor().mutable_data();
	size_t type_size = weights.h_tensor().get_dtype_size();
    if (!conv_bias_term) {
        bias.re_alloc(Shape({1,batchnorm_mean.size(),1,1}));
        void* new_bias_data = bias.h_tensor().mutable_data();
        memset(new_bias_data, 0, type_size * bias.h_tensor().size());
    }
    float* bias_p = (float*)bias.h_tensor().mutable_data();
    batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
    int chw = c * h * w;
    for (int i = 0; i < n; i++) {
        float alpha = 1.f;
        float beta = 0.f;
        // insert batchnorm parameters
        alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
        alpha = 1.f / sqrtf(alpha);
        beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
        beta = beta * alpha;
        for (int j = 0; j < chw; j++) {
            weights_p[i * chw + j] *= alpha;
        }
        bias_p[i] *= alpha;
        bias_p[i] += beta;
    }
}

std::vector<float> get_scale_basic(const float* in_data, int axis_size, \
								   long long outer_size, long long inner_size , float scale_factor) {
	std::vector<float> scale_out(axis_size);
	for (int c = 0; c < axis_size; ++c) {
		float max_val = 0.f;
		const float* din = in_data + c * inner_size;
		for (int j = 0; j < outer_size; ++j) {
			const float* ptr_in = din + j * inner_size * axis_size;
			for (int i = 0; i < inner_size; ++i) {
				float read_data = fabsf(ptr_in[i]);
				max_val = (read_data > max_val) ? read_data : max_val;
			}
		}
		scale_out[c] = max_val / scale_factor;
	}
	return scale_out;
}

void fp32_to_int8_basic(const float* din, char* dout, const float* scale, \
						int axis_size, long long outer_size, long long inner_size) {
	int loop_size = axis_size * outer_size;
	for (int i = 0; i < loop_size; ++i) {
		float inv_scale = 1.f / scale[i % axis_size];
		for (int j = 0; j < inner_size; ++j) {
			dout[j] = static_cast<char>(roundf(din[j] * inv_scale));
		}
		dout += inner_size;
		din += inner_size;
	}
}

void int8_to_fp32_basic(const char* din, float* dout, const float* scale, \
						int axis_size, long long outer_size, long long inner_size) {
	int loop_size = axis_size * outer_size;
	for (int i = 0; i < loop_size; ++i) {
		float scale_in = scale[i % axis_size];
		for (int j = 0; j < inner_size; ++j) {
			dout[j] = din[j] * scale_in;
		}
		dout += inner_size;
		din += inner_size;
	}
}

bool trans_weights_dtype_basic(Tensor<X86>& weights, DataType type, float scale_factor, \
							bool is_trans = false) {

	if (weights.get_dtype() == type) {
		return true;
	}
	if (type == AK_FLOAT && weights.get_dtype() == AK_INT8) {
		//! trans int8 weights to fp32 weights
		if (weights.get_scale().size() <= 0) {
			LOG(ERROR) << "ERROR: Trans weights from int8 to fp32, without scale";
			return false;
		}
		Tensor<X86> tmp_tensor;
		tmp_tensor.re_alloc(weights.valid_shape(), AK_FLOAT);
		std::vector<float> scale = weights.get_scale();
		const char* din = static_cast<const char*>(weights.data());
		float* dout = static_cast<float*>(tmp_tensor.mutable_data());
		if (is_trans) {
			//! for deconv
			int axis_size = weights.valid_shape()[0];
			int outer_size = weights.valid_shape()[1];
			int inner_size = weights.valid_shape()[2] * weights.valid_shape()[3];
			int8_to_fp32_basic(din, dout, scale.data(), axis_size, outer_size, inner_size);
		} else {
			//! for conv
			int axis_size = weights.valid_shape()[0];
			int outer_size = 1;
			int inner_size = weights.count_valid(1, weights.dims());
			int8_to_fp32_basic(din, dout, scale.data(), axis_size, outer_size, inner_size);
		}
		weights.re_alloc(weights.valid_shape(), AK_FLOAT);
		weights.copy_from(tmp_tensor);
	} else if (type == AK_INT8 && weights.get_dtype() == AK_FLOAT) {
		//! trans fp32 weights to int8 weights
		Tensor<X86> tmp_tensor;
		tmp_tensor.re_alloc(weights.valid_shape(), AK_INT8);
		std::vector<float> scale;
		const float* din = static_cast<const float*>(weights.data());
		char* dout = static_cast<char*>(tmp_tensor.mutable_data());
		if (is_trans) {
			//! for deconv, chout and chin in inversed
			//! real layout is: chin, chout, kh, kw
			int axis_size = weights.valid_shape()[0];
			int outer_size = weights.valid_shape()[1];
			int inner_size = weights.valid_shape()[2] * weights.valid_shape()[3];
			scale = get_scale_basic(din, axis_size, outer_size, inner_size, scale_factor);
			fp32_to_int8_basic(din, dout, scale.data(), axis_size, outer_size, inner_size);
		} else {
			//! for conv
			//! layout is: chout, chin, kh, kw
			int axis_size = weights.valid_shape()[0];
			int inner_size = weights.valid_size() / axis_size;
			scale = get_scale_basic(din, axis_size, 1, inner_size, scale_factor);
			fp32_to_int8_basic(din, dout, scale.data(), axis_size, 1, inner_size);
		}
		//! set weights scale
		weights.set_scale(scale);
		weights.re_alloc(weights.valid_shape(), AK_INT8);
		weights.copy_from(tmp_tensor);
	} else {
		LOG(ERROR) << "ERROR: Trans weights fialed, unsupported data type";
		return false;
	}
	return true;
}

void trans_conv_weights_inplace(Tensor<X86>& th, DataType op_precision, bool lite_mode){

    auto scale = th.get_scale();
    if (scale.size() == 1){
        float scale_old = scale[0];
        scale.resize(th.valid_shape()[0]);
        for (int i = 0; i < scale.size(); ++i){
            scale[i] = scale_old;
        }
        th.set_scale(scale);
    }
    if (lite_mode) {
        trans_weights_dtype_basic(th, AK_INT8, 127.0f, false);
        return;
    }

    if (th.get_dtype() == AK_INT8 && op_precision == AK_INT8){
		return;
	}
	if (th.get_dtype() == AK_FLOAT && op_precision == AK_INT8) {
        trans_weights_dtype_basic(th, AK_INT8, 127.0f, false);
		return;
	}
	// trans weights to fp32 when op_precision is fp32
	if (th.get_dtype() == AK_INT8 && op_precision == AK_FLOAT){
        trans_weights_dtype_basic(th, AK_FLOAT, 127.0f, false);
		return;
	}
	// op_precision == AK_FLOAT && th.dtype == AK_FLOAT
    if (th.get_dtype() == AK_FLOAT && op_precision == AK_FLOAT){
	    const float* din = (float*)th.data();
        int axis_size = th.valid_shape()[0];
        int outer_size = 1;
        int inner_size = th.count_valid(1, th.dims());
        std::vector<float> scale = get_scale_basic(din, axis_size, outer_size, inner_size, 127.0f);
        th.set_scale(scale);
        return;
	}
	return;
}

void trans_deconv_weights_inplace(Tensor<X86>& th, DataType op_precision, bool lite_mode){
    if (th.get_dtype() == AK_INT8 && op_precision == AK_INT8){
        return;
    }
    if (th.get_dtype() == AK_FLOAT && (op_precision == AK_INT8 || lite_mode)) {
        trans_weights_dtype_basic(th, AK_INT8, 127.0f, true);
        return;
    }
    // trans weights to fp32 when op_precision is fp32
    if (th.get_dtype() == AK_INT8 && op_precision == AK_FLOAT){
        trans_weights_dtype_basic(th, AK_FLOAT, 127.0f, true);
        return;
    }
    // op_precision == AK_FLOAT && th.dtype == AK_FLOAT
    if (th.get_dtype() == AK_FLOAT && op_precision == AK_FLOAT){
        const float* din = (float*)th.data();
        int axis_size = th.valid_shape()[0];
        int outer_size = th.valid_shape()[1];
        int inner_size = th.valid_shape()[2] * th.valid_shape()[0];
        std::vector<float> scale = get_scale_basic(din, axis_size, outer_size, inner_size, 127.0f);
        th.set_scale(scale);
        return;
    }
    return;
}
} /* namespace lite */

} /* namespace anakin */

#endif
