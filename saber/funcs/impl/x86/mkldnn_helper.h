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

#ifndef ANAKIN_SABER_MKLDNN_HELPER_H
#define ANAKIN_SABER_MKLDNN_HELPER_H

#include "anakin_config.h"
#include "saber/core/common.h"
#include "saber/saber_types.h"
#include "saber/core/tensor.h"

#include "mkldnn.hpp"

namespace anakin{

namespace saber{                                             

typedef mkldnn::memory::data_type    mkldnn_mem_dtype;
typedef mkldnn::memory::format       mkldnn_mem_format;
typedef mkldnn::memory::dims         mkldnn_mem_dim;
typedef mkldnn::memory               mkldnn_mem;
typedef std::shared_ptr<mkldnn_mem>  mkldnn_mem_ptr;
typedef mkldnn::deconvolution_forward mkldnn_deconv;
typedef mkldnn::convolution_forward  mkldnn_conv;
typedef mkldnn::eltwise_forward         mkldnn_relu;

template <typename T>
using desc = typename T::desc;
template <typename T>
using pdesc = typename T::primitive_desc;

mkldnn_mem_format get_mkldnn_format(LayoutType layout);
mkldnn_mem_format get_mkldnn_format(LayoutType in_layout, LayoutType out_layout);
mkldnn_mem_dtype get_mkldnn_dtype(DataType dtype);

desc<mkldnn_mem> create_mkldnn_memory_desc(
                    const std::vector<int>& dims,
                    mkldnn_mem_dtype dtype, 
                    mkldnn_mem_format layout);

template <DataType Dtype>
desc<mkldnn_mem> create_mkldnn_memory_desc(const std::vector<int>& sh,
                                  mkldnn_mem_format fmt = mkldnn_mem_format::any){
  mkldnn_mem_dim tz = sh;
  mkldnn_mem_dtype dt = get_mkldnn_dtype(Dtype);
  return desc<mkldnn_mem>({tz}, dt, fmt);
}

mkldnn_mem_ptr create_mkldnn_memory(Tensor<X86>* tensor, mkldnn::engine e);

mkldnn_mem_ptr create_mkldnn_memory(Tensor<X86>* tensor, const std::vector<int>& sh, mkldnn::engine e);

mkldnn_mem_ptr create_mkldnn_memory(Tensor<X86>* tensor, const std::vector<int>& sh, 
    mkldnn_mem_format mft, mkldnn_mem_dtype dt, mkldnn::engine e);

mkldnn_mem_ptr create_mkldnn_memory_no_data(const Tensor<X86>* tensor, mkldnn::engine e);


} // namespace mkldnn
} // namespace anakin

#endif //SABER_MKLDNN_HELPER_H
