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

#ifndef ANAKIN_SABER_LITE_CORE_TENSOR_OP_H
#define ANAKIN_SABER_LITE_CORE_TENSOR_OP_H

#include "saber/lite/core/tensor_lite.h"

namespace anakin{

namespace saber{

namespace lite{

const float eps = 1e-6f;

/**
 *  \brief Fill the host tensor buffer with rand value.
 *  \param tensor  The reference of input tensor.
 */
template <ARMType ttype, DataType dtype>
void fill_tensor_const(Tensor<ttype, dtype>& tensor, typename DataTrait<ttype, dtype>::Dtype value);


/**
 *  \brief Fill the host tensor buffer with rand value.
 *  \param The reference of input tensor.
 */
template <ARMType ttype, DataType dtype>
void fill_tensor_rand(Tensor<ttype, dtype>& tensor);


/**
 *  \brief Fill the host tensor buffer with rand value from vstart to vend.
 *  \param tensor The reference of input tensor.
 */
template <ARMType ttype, DataType dtype>
void fill_tensor_rand(Tensor<ttype, dtype>& tensor, \
    typename DataTrait<ttype, dtype>::Dtype vstart, \
    typename DataTrait<ttype, dtype>::Dtype vend);

/**
 *  \brief Print the data in host tensor.
 *  \param tensor  The reference of input tensor.
 */
template <ARMType ttype, DataType dtype>
void print_tensor(Tensor<ttype, dtype>& tensor);

template <ARMType ttype, DataType dtype>
void print_tensor_valid(Tensor<ttype, dtype>& tensor);

template <typename Dtype>
void tensor_cmp_host(const Dtype* src1, const Dtype* src2, int size, double& max_ratio, double& max_diff);

} //namespace lite

} // namespace saber

} // namespace anakin

#endif //ANAKIN_SABER_LITE_CORE_TENSOR_OP_H
