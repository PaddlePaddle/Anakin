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

template<typename D, typename T>
class WeightsFusion{
public:
    WeightsFusion(){};
    /**
     * \brief  update conv weights with batchnorm and scale parameters.
     */
    static void update_weights(PBlock<T> weights, PBlock<T> bias,
                               int n, int c, int h, int w, bool conv_bias_term,
                               float batchnorm_scale, float batchnorm_eps,
                               std::vector<float> batchnorm_mean,
                               std::vector<float> batchnorm_variance,
                               std::vector<float> scale_w,
                               std::vector<float> scale_b,
                               bool scale_bias_term){
        LOG(ERROR) << "unsupport weights dtype";
    }

    /**
     * \brief  update conv weights with affine channel parameters.
     */
    static void update_conv_affine_channel_weights(PBlock<T> weights, PBlock<T> bias,
                                            int n, int c, int h, int w,
                                            std::vector<float> affine_channel_w,
                                            std::vector<float> affine_channel_b){
        LOG(ERROR) << "support weights dtype";
    };

    /**
     * \brief  update conv weights with batchnorm.
     */
    static void update_weights_without_scale(PBlock<T> weights, PBlock<T> bias,
                                      int n, int c, int h, int w, bool conv_bias_term,
                                      float batchnorm_scale, float batchnorm_eps,
                                      std::vector<float> batchnorm_mean,
                                      std::vector<float> batchnorm_variance){
        LOG(ERROR) << "support weights dtype";
    }

    /**
     * \brief  update conv weights with batchnorm and scale parameters.
     */
    static void update_deconv_weights(PBlock<T> weights, PBlock<T> bias,
                               int n, int c, int h, int w, bool conv_bias_term,
                               float batchnorm_scale, float batchnorm_eps,
                               std::vector<float> batchnorm_mean,
                               std::vector<float> batchnorm_variance,
                               std::vector<float> scale_w,
                               std::vector<float> scale_b,
                               bool scale_bias_term){
        LOG(ERROR) << "support weights dtype";
    }

    /**
     * \brief  update conv weights with batchnorm.
     */
    static void update_deconv_weights_without_scale(PBlock<T> weights, PBlock<T> bias,
                                             int n, int c, int h, int w, bool conv_bias_term,
                                             float batchnorm_scale, float batchnorm_eps,
                                             std::vector<float> batchnorm_mean,
                                             std::vector<float> batchnorm_variance){
        LOG(ERROR) << "support weights dtype";
    };
};

template<typename T>
class WeightsFusion<float, T>{
public:
    WeightsFusion(){};
    /**
     * \brief  update conv weights with batchnorm and scale parameters.
     */
    static void update_weights(PBlock<T> weights, PBlock<T> bias,
                               int n, int c, int h, int w, bool conv_bias_term,
                               float batchnorm_scale, float batchnorm_eps,
                               std::vector<float> batchnorm_mean,
                               std::vector<float> batchnorm_variance,
                               std::vector<float> scale_w,
                               std::vector<float> scale_b,
                               bool scale_bias_term);

    /**
     * \brief  update conv weights with affine channel parameters.
     */
    static void update_conv_affine_channel_weights(PBlock<T> weights, PBlock<T> bias,
                                            int n, int c, int h, int w,
                                            std::vector<float> affine_channel_w,
                                            std::vector<float> affine_channel_b);

    /**
     * \brief  update conv weights with batchnorm.
     */
    static void update_weights_without_scale(PBlock<T> weights, PBlock<T> bias,
                                      int n, int c, int h, int w, bool conv_bias_term,
                                      float batchnorm_scale, float batchnorm_eps,
                                      std::vector<float> batchnorm_mean,
                                      std::vector<float> batchnorm_variance);

    /**
     * \brief  update conv weights with batchnorm and scale parameters.
     */
    static void update_deconv_weights(PBlock<T> weights, PBlock<T> bias,
                               int n, int c, int h, int w, bool conv_bias_term,
                               float batchnorm_scale, float batchnorm_eps,
                               std::vector<float> batchnorm_mean,
                               std::vector<float> batchnorm_variance,
                               std::vector<float> scale_w,
                               std::vector<float> scale_b,
                               bool scale_bias_term);

    /**
     * \brief  update conv weights with batchnorm.
     */
    static void update_deconv_weights_without_scale(PBlock<T> weights, PBlock<T> bias,
                                             int n, int c, int h, int w, bool conv_bias_term,
                                             float batchnorm_scale, float batchnorm_eps,
                                             std::vector<float> batchnorm_mean,
                                             std::vector<float> batchnorm_variance);
};

template<typename T>
class WeightsFusion<char, T>{
public:
    WeightsFusion(){};
    /**
    * \brief  update conv weights with batchnorm and scale parameters.
	*/
	static void update_weights(PBlock<T> weights, PBlock<T> bias,
							   int n, int c, int h, int w, bool conv_bias_term,
							   float batchnorm_scale, float batchnorm_eps,
							   std::vector<float> batchnorm_mean,
							   std::vector<float> batchnorm_variance,
							   std::vector<float> scale_w,
							   std::vector<float> scale_b,
							   bool scale_bias_term);

    /**
	 * \brief  update conv weights with affine channel parameters.
    */
	static void update_conv_affine_channel_weights(PBlock<T> weights, PBlock<T> bias,
											int n, int c, int h, int w,
											std::vector<float> affine_channel_w,
											std::vector<float> affine_channel_b);

	/**
	 * \brief  update conv weights with batchnorm.
	 */
	static void update_weights_without_scale(PBlock<T> weights, PBlock<T> bias,
									  int n, int c, int h, int w, bool conv_bias_term,
									  float batchnorm_scale, float batchnorm_eps,
									  std::vector<float> batchnorm_mean,
									  std::vector<float> batchnorm_variance);

	/**
	 * \brief  update conv weights with batchnorm and scale parameters.
	 */
	static void update_deconv_weights(PBlock<T> weights, PBlock<T> bias,
							   int n, int c, int h, int w, bool conv_bias_term,
							   float batchnorm_scale, float batchnorm_eps,
							   std::vector<float> batchnorm_mean,
							   std::vector<float> batchnorm_variance,
							   std::vector<float> scale_w,
							   std::vector<float> scale_b,
							   bool scale_bias_term);

	/**
	 * \brief  update conv weights with batchnorm.
	 */
	static void update_deconv_weights_without_scale(PBlock<T> weights, PBlock<T> bias,
											 int n, int c, int h, int w, bool conv_bias_term,
											 float batchnorm_scale, float batchnorm_eps,
											 std::vector<float> batchnorm_mean,
											 std::vector<float> batchnorm_variance);
};


} /* namespace anakin */

#endif
