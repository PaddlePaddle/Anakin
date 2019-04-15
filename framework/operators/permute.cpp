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
#include "framework/operators/permute.h"

namespace anakin {

namespace ops {

#define INSTANCE_PERMUTE(Ttype, Ptype) \
template<> \
void Permute<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
        std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<PermuteHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<PermuteHelper<Ttype, Ptype>*>\
                  (this->_helper)->_param_permute; \
    impl->_funcs_permute(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status PermuteHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << " Parsing Permute op parameter.";
    auto dims = GET_PARAMETER(PTuple<int>, dims);

    for (int i = 0; i < dims.size(); i++) {
        DLOG(INFO) << " |-- dims [" << i << "]: " << dims[i];
    }

    saber::PermuteParam<Ttype> permute_param(dims.vector());
    _param_permute = permute_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PermuteHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_permute.init(ins, outs, _param_permute, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PermuteHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_permute.compute_output_shape(ins, outs, _param_permute));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_PERMUTE(NV, Precision::FP32);
template class PermuteHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Permute, PermuteHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_PERMUTE(AMD, Precision::FP32);
template class PermuteHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Permute, PermuteHelper, AMD, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_PERMUTE(X86, Precision::FP32);
template class PermuteHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Permute, PermuteHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_PERMUTE(ARM, Precision::FP32);
template class PermuteHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Permute, PermuteHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Permute)
.Doc("Permute operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("permute")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("permute")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("permute")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("permute")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims for permuting the order of input ");

} /* namespace ops */

} /* namespace anakin */


