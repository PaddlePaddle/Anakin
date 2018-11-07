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

#ifndef ANAKIN_FRAMEWORK_UTILS_PARAMETER_FUSION_H
#define ANAKIN_FRAMEWORK_UTILS_PARAMETER_FUSION_H

#include <string>
#include <unordered_map>
#include "framework/core/parameter.h"

namespace anakin {

/**
 * \brief  update conv weights with batchnorm and scale parameters.
 */
template<typename D, typename T>
void update_weights(PBlock<T> weights, PBlock<T> bias,
					int n, int c, int h, int w, bool conv_bias_term, 
					float batchnorm_scale, float batchnorm_eps, 
					std::vector<float> batchnorm_mean, 
					std::vector<float> batchnorm_variance, 
					std::vector<float> scale_w, 
					std::vector<float> scale_b, 
					bool scale_bias_term) {
	D* weights_p = (D* )(weights.h_tensor().mutable_data());
	if(!conv_bias_term) {
		bias.re_alloc(Shape4d({1, batchnorm_mean.size(), 1, 1}));
		void* new_bias_data = bias.h_tensor().mutable_data();
		memset(new_bias_data, 0, sizeof(D) * bias.h_tensor().size());
	}
	D* bias_p = (D* )(bias.h_tensor().mutable_data());

	batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
	int chw = c * h * w;
	for (int i = 0; i < n; i++) {
		D alpha = 1.f;
		D beta = 0.f;
		// insert batchnorm parameters
		alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
		alpha = 1.f / sqrtf(alpha);
		beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
		beta = beta * alpha;

		// insert scale parameters
		alpha = scale_w[i] * alpha;
		if(scale_bias_term) {
			beta = beta * scale_w[i] + scale_b[i];
		} else {
			beta = beta * scale_w[i];
		}
		for(int j=0; j < chw; j++) {
			weights_p[i * chw + j] *= alpha;
		}
		bias_p[i] *= alpha;
		bias_p[i] += beta;
	}
    weights.d_tensor().copy_from(weights.h_tensor());
    bias.d_tensor().copy_from(bias.h_tensor());
}

/**
 * \brief  update conv weights with batchnorm.
 */
template<typename D, typename T>
void update_weights_without_scale(PBlock<T> weights, PBlock<T> bias,
					int n, int c, int h, int w, bool conv_bias_term, 
					float batchnorm_scale, float batchnorm_eps, 
					std::vector<float> batchnorm_mean, 
					std::vector<float> batchnorm_variance) {
	D* weights_p = (D* )(weights.h_tensor().mutable_data());
	if(!conv_bias_term) {
		bias.re_alloc(Shape4d({1, batchnorm_mean.size(), 1, 1}));
		void* new_bias_data = bias.h_tensor().mutable_data();
		memset(new_bias_data, 0, sizeof(D) * bias.h_tensor().size());
	}
	D* bias_p = (D* )(bias.h_tensor().mutable_data());

	batchnorm_scale = (batchnorm_scale == 0) ? 1.f : 1.f / batchnorm_scale;
	int chw = c * h * w;
	for (int i = 0; i < n; i++) {
		D alpha = 1.f;
		D beta = 0.f;
		// insert batchnorm parameters
		alpha = batchnorm_variance[i] * batchnorm_scale + batchnorm_eps;
		alpha = 1.f / sqrtf(alpha);
		beta = -1.f * (batchnorm_mean[i] * batchnorm_scale);
		beta = beta * alpha;
		for(int j=0; j < chw; j++) {
			weights_p[i * chw + j] *= alpha;
		}
		bias_p[i] *= alpha;
		bias_p[i] += beta;
	}
    weights.d_tensor().copy_from(weights.h_tensor());
    bias.d_tensor().copy_from(bias.h_tensor());
}
} /* namespace anakin */

#endif
