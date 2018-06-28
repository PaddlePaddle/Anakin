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

namespace anakin {

namespace lite {

/**
 * \brief  update conv weights with batchnorm and scale parameters.
 */
template<typename D, typename T>
void update_weights(PBlock<D, T> weights, PBlock<D, T> bias,
					int num, int c, int h, int w, bool conv_bias_term, 
					float batchnorm_scale, float batchnorm_eps, 
					std::vector<float> batchnorm_mean, 
					std::vector<float> batchnorm_variance, 
					std::vector<float> scale_w, 
					std::vector<float> scale_b, 
					bool scale_bias_term) {
	D* weights_p = weights.h_tensor().mutable_data();
	D* bias_p = conv_bias_term ? bias.h_tensor().mutable_data() : nullptr;
	int chw = c*h*w;
	for(int i=0; i <n; i++ ) {
		D alpha = 0.f;
		D beta = 0.f;
		batchnorm_scale = (batchnorm_scale == 0) ? 1.f : batchnorm_scale;
		for(int j=0; j < chw; j++) {
		}
	}
}

} /* namespace lite */

} /* namespace anakin */

#endif
