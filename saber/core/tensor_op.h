/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_TENSOR_OP_H
#define ANAKIN_SABER_TENSOR_OP_H

#include "core/tensor.h"
#include "context.h"
#include "anakin_config.h"

namespace anakin{

namespace saber{

const float eps = 1e-6f;

/**
* \brief reorder reorder tensors from src layout to dst layout
* \param src  source tensor reference
* \param dst  destination tensor reference
*/
template <class Tensor_s, class Tensor_d>
void reorder(Tensor_s& src, Tensor_d& dst);

/**
 *  \brief Fill the tensor buffer with rand value.
 *  \param tensor  The reference of input tensor.
 */
template <typename TargetType>
void fill_tensor_const(Tensor<TargetType>& tensor, float value, typename Tensor<TargetType>::API::stream_t stream = NULL);

/**
 *  \brief Fill the tensor buffer with rand value.
 *  \param The reference of input tensor.
 */
template <typename TargetType>
void fill_tensor_seq(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream = NULL);

/**
 *  \brief Fill the tensor buffer with rand value.
 *  \param The reference of input tensor.
 */
template <typename TargetType>
void fill_tensor_rand(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream = NULL);


/**
 *  \brief Fill the tensor buffer with rand value from vstart to vend.
 *  \param tensor The reference of input tensor.
 */
template <typename TargetType>
void fill_tensor_rand(Tensor<TargetType>& tensor, float vstart, float vend, typename Tensor<TargetType>::API::stream_t stream = NULL);

/**
 *  \brief Print the data in host tensor.
 *  \param tensor  The reference of input tensor.
 */
template <typename TargetType>
void print_tensor(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream = NULL);

/**
 *  \brief Print the valid data in host tensor.
 *  \param tensor  The reference of input tensor.
 */
template <typename TargetType>
void print_tensor_valid(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream = NULL);

/**
 *  \brief compute mean value of the valid data in device tensor.
 *  \param tensor  The reference of input tensor.
 */
template <typename TargetType>
double tensor_mean_value(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream = NULL);

/**
 *  \brief compute mean value of the valid data in device tensor.
 *  \param tensor  The reference of input tensor.
 */
template <typename TargetType>
double tensor_mean_value_valid(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream = NULL);

template <typename Dtype >
void tensor_cmp_host(const Dtype* src1, const Dtype* src2, int size, double& max_ratio, double& max_diff);

#ifdef USE_CUDA

/// This transform helper is only used to transform inputs or outputs,
/// This is NOT mean to transform wegiths.

class DataTensorTransformHelper{
public:

    DataTensorTransformHelper(float scale = 0.f)
        : _in_scale(scale)
        , _weight_scale(NULL)
    {}

    ~DataTensorTransformHelper() {
        if (_weight_scale != NULL) {
            CUDA_CHECK(cudaFree(_weight_scale));
        }
    }

    SaberStatus init(float scale, std::vector<float> weight_scale, Context<NV > &ctx){
        _vector_weight_scale = weight_scale;
        unsigned long weight_size = weight_scale.size();
        CHECK_GT(scale, 0);
        CHECK_GT(weight_size, 0);

        _in_scale = scale;

        for (int i = 0; i < weight_size; ++i) {
            weight_scale[i] *= scale;
        }
        CUDA_CHECK(cudaMalloc(&_weight_scale, sizeof(float) * weight_size));
        CUDA_CHECK(cudaMemcpy(_weight_scale, &weight_scale[0],
                              sizeof(float) * weight_size, cudaMemcpyHostToDevice));
        return SaberSuccess;
    }

    SaberStatus init(Context<NV > &ctx){
        unsigned long weight_size = _vector_weight_scale.size();
        CHECK_GT(_in_scale, 0);
        CHECK_GT(weight_size, 0);

        for (int i = 0; i < weight_size; ++i) {
            _vector_weight_scale[i] *= _in_scale;
        }
        CUDA_CHECK(cudaMalloc(&_weight_scale, sizeof(float) * weight_size));
        CUDA_CHECK(cudaMemcpy(_weight_scale, &_vector_weight_scale[0],
                              sizeof(float) * weight_size, cudaMemcpyHostToDevice));
        return SaberSuccess;
    }

    template <typename outTensor, typename inTensor>
    SaberStatus transform(outTensor &out_tensor, const inTensor &in_tensor, Context<NV> ctx);

    template <typename outTensor, typename inTensor>
    SaberStatus convert_weights(outTensor &out_tensor, const inTensor &in_tensor, Context<NV> ctx);

    template <typename outTensor, typename inTensor>
    SaberStatus convert_bias(outTensor &out_tensor, const inTensor &in_tensor, Context<NV> ctx);
private:
    float _in_scale;
    float* _weight_scale;
    std::vector<float> _vector_weight_scale;
};

#endif

} // namespace saber

} // namespace anakin

#endif //ANAKIN_SABER_TENSOR_OP_H
