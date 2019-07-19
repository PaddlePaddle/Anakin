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
#include "framework/operators/one_hot.h"

namespace anakin {

namespace ops {

#define INSTANCE_ONE_HOT(Ttype, Ptype) \
template<> \
void OneHot<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<OneHotHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<OneHotHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_one_hot; \
    impl->_funcs_one_hot(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status OneHotHelper<Ttype, Ptype>::InitParam() {

    DLOG(WARNING) << "Parsing OneHot op parameter.";
    auto depth = GET_PARAMETER(int, depth);
    saber::OneHotParam<Ttype> one_hot_param(depth);
    _param_one_hot = one_hot_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status OneHotHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    //different device pleace change here..
    saber::ImplEnum impl_e = SABER_IMPL;
    SABER_CHECK(_funcs_one_hot.init(ins, outs, _param_one_hot, SPECIFY, impl_e, ctx));

    // check if weights have been transposed
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status OneHotHelper<Ttype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    SABER_CHECK(_funcs_one_hot.compute_output_shape(ins, outs, _param_one_hot));
    return Status::OK();
}

#ifdef USE_CUDA
template class OneHotHelper<NV, Precision::FP32>;
INSTANCE_ONE_HOT(NV, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(OneHot, OneHotHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ONE_HOT(X86, Precision::FP32);
template class OneHotHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(OneHot, OneHotHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ONE_HOT(ARM, Precision::FP32);
template class OneHotHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(OneHot, OneHotHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(OneHot)
.Doc("OneHot operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("one_hot")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("one_hot")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("one_hot")
#endif
.num_in(1)
.num_out(1)
.Args<int>("depth", " depth of one_hot ");
} /* namespace ops */

} /* namespace anakin */


