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
#include "framework/operators/concat.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
Status ConcatHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Concat op parameter.";
    auto axis = GET_PARAMETER(int, axis);
    ConcatParam<Ttype> param_concat(axis);
    _param_concat = param_concat;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConcatHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx,
                                  const std::vector<Tensor4dPtr<Ttype> >& ins,
                                    std::vector<Tensor4dPtr<Ttype> >& outs){
    SABER_CHECK(_funcs_concat.init(ins, outs, _param_concat, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConcatHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                                std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_concat.compute_output_shape(ins, outs, _param_concat));
    return Status::OK();
}


#define INSTANCE_CONCAT(Ttype, Ptype) \
template<> \
void Concat<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
                std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<ConcatHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ConcatHelper<Ttype, Ptype>*>(this->_helper)->_param_concat; \
    impl->_funcs_concat(ins, outs, param, ctx); \
}

#ifdef USE_CUDA
INSTANCE_CONCAT(NV, Precision::FP32);
template class ConcatHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_CONCAT(AMD, Precision::FP32);
template class ConcatHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, AMD, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONCAT(ARM, Precision::FP32);
template class ConcatHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, ARM, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_CONCAT(X86, Precision::FP32);
template class ConcatHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Concat)
.Doc("Concat operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("concat")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("concat")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("concat")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("concat")
#endif
.num_in(2)
.num_out(1)
.Args<int>("axis", " axis for concat the input ");

} /* namespace ops */

} /* namespace anakin */


