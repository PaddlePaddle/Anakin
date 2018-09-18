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

#include "core/tensor_op.h"
namespace anakin {

namespace saber {

template<>
void fill_tensor_const<AMD>(Tensor<AMD>& tensor, float value,
                           typename Tensor<AMD>::API::stream_t stream = NULL) {
    Tensor<AMDHX86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    fill_tensor_const(temp_tensor, value);
    tensor.copy_from(temp_tensor);
}
template<>
void fill_tensor_rand<AMD>(Tensor<AMD>& tensor, typename Tensor<AMD>::API::stream_t stream = NULL) {
    Tensor<AMDHX86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    fill_tensor_rand(temp_tensor);
    tensor.copy_from(temp_tensor);
}

template<>
void fill_tensor_rand<AMD>(Tensor<AMD>& tensor, float vstart, float vend,
                          typename Tensor<AMD>::API::stream_t stream = NULL) {
    Tensor<AMDHX86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    fill_tensor_rand(temp_tensor, vstart, vend);
    tensor.copy_from(temp_tensor);
}

template<>
void print_tensor<AMD>(Tensor<AMD>& tensor, typename Tensor<AMD>::API::stream_t stream = NULL) {
    LOG(INFO) << "device tensor data";
    Tensor<AMDHX86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    temp_tensor.copy_from(tensor);
    print_tensor(temp_tensor);
}

template<>
void print_tensor_valid<AMD>(Tensor<AMD>& tensor, typename Tensor<AMD>::API::stream_t stream = NULL) {
    LOG(INFO) << "device tensor data";
    print_tensor(tensor);
}

template<>
double tensor_mean_value<AMD>(Tensor<AMD>& tensor, typename Tensor<AMD>::API::stream_t stream = NULL) {
    Tensor<AMDHX86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    temp_tensor.copy_from(tensor);
    return tensor_mean_value(temp_tensor);
}

template<>
double tensor_mean_value_valid<AMD>(Tensor<AMD>& tensor,
                                   typename Tensor<AMD>::API::stream_t stream = NULL) {
    Tensor<AMDHX86> temp_tensor(tensor.valid_shape(),tensor.get_dtype());
    temp_tensor.copy_from(tensor);
    return tensor_mean_value(temp_tensor);
}
}
}