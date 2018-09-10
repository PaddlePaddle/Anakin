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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CRFDECODING_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CRFDECODING_H

#include "saber/funcs/impl/impl_crf_decoding.h"
#include "saber/saber_funcs_param.h"

namespace anakin{
namespace saber {

template <DataType OpDtype>
class SaberCrfDecoding<NV, OpDtype> : public ImplBase<
        NV, OpDtype,
        CrfDecodingParam<NV> >
{
    public:
    typedef typename DataTrait< NV, OpDtype>::Dtype OpDataType;

    SaberCrfDecoding() = default;

    ~SaberCrfDecoding() {}

    virtual SaberStatus init(const std::vector<Tensor< NV> *>& inputs,
                             std::vector<Tensor< NV> *>& outputs,
                             CrfDecodingParam< NV> &param,
                             Context< NV> &ctx){
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor< NV> *>& inputs,
                               std::vector<Tensor< NV> *>& outputs,
                               CrfDecodingParam< NV> &param,
                               Context< NV> &ctx){
        CHECK_EQ(inputs[0]->get_dtype(), OpDtype) << "inputs data type should be same with OpDtype";
        CHECK_EQ(outputs[0]->get_dtype(), OpDtype) << "outputs data type should be same with OpDtype";
    
        _track.re_alloc(inputs[0]->valid_shape(), AK_INT32);
        _alpha.re_alloc(inputs[0]->valid_shape(), OpDtype);
        
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor< NV> *>& inputs,
                                 std::vector<Tensor< NV> *>& outputs,
                                 CrfDecodingParam< NV> &param) override;
private:
    Tensor< NV> _alpha;
    Tensor< NV> _track;
    Tensor< NV> _seq;
    int _aligned_tag_num;
};
}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CRFDECODING_H