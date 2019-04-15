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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CONCAT_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CONCAT_H

#include "anakin_config.h"
#include "saber/funcs/impl/impl_concat.h"
#include "saber/core/tensor.h"

#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberConcat<X86, OpDtype> : \
    public ImplBase<
        X86,
        OpDtype,
        ConcatParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberConcat() : _num_concats(0), _concat_input_size(0),
                    dst_data_(nullptr),
                    srcs_data_(nullptr), src_with_offset_(nullptr),
                    tail_(nullptr), ic_(nullptr),
                    nb_ic_(nullptr), scale_(nullptr),
                    block_(nullptr){

    };
    ~SaberConcat() {

        if (srcs_data_ != nullptr) {
            delete srcs_data_;
            srcs_data_ = nullptr;
        }
        if (src_with_offset_ != nullptr) {
            delete src_with_offset_;
            src_with_offset_ = nullptr;
        }
        if (tail_ != nullptr) {
            delete tail_;
            tail_ = nullptr;
        }
        if (ic_ != nullptr) {
            delete ic_;
            ic_ = nullptr;
        }
        if (nb_ic_ != nullptr) {
            delete nb_ic_;
            nb_ic_ = nullptr;
        }
        if (scale_ != nullptr) {
            delete scale_;
            scale_ = nullptr;
        }
        if (block_ != nullptr) {
            delete block_;
            block_ = nullptr;
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                      std::vector<Tensor<X86>*>& outputs,
                      ConcatParam<X86> &param, Context<X86> &ctx){
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                        std::vector<Tensor<X86>*>& outputs,
                        ConcatParam<X86> &param, Context<X86> &ctx)override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                          std::vector<Tensor<X86>*>& outputs,
                          ConcatParam<X86> &param)override;


private:
    int _num_concats;
    int _concat_input_size;

    unsigned long* tail_;
    unsigned int* ic_;
    unsigned int* nb_ic_;
    unsigned int* block_;
    float* scale_;
    unsigned char* dst_data_;
    const unsigned char** srcs_data_;
    const unsigned char** src_with_offset_;
    virtual SaberStatus init_conf(jit::jit_concat_conf_t &jpp,
                                  const std::vector<Tensor<X86>*> &inputs,
                                  std::vector<Tensor<X86>*> &outputs,
                                  ConcatParam<X86> &param);

    virtual SaberStatus check_conf(const jit::jit_concat_conf_t &jpp,
                                   const std::vector<Tensor<X86>*> &inputs,
                                   std::vector<Tensor<X86>*> &outputs,
                                   ConcatParam<X86> &param);
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CONCAT_H
