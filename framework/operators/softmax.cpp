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

#include "framework/operators/softmax.h"

namespace anakin {

namespace ops {

#define INSTANCE_SOFTMAX(Ttype, Ptype) \
template<> \
void Softmax<Ttype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<SoftmaxHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<SoftmaxHelper<Ttype, Ptype>*>\
                  (this->_helper)->_param_softmax; \
    impl->_funcs_softmax(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status SoftmaxHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Softmax op parameter.";
    auto axis = GET_PARAMETER(int, axis);

    SoftmaxParam<Ttype> param_softmax(axis);
    _param_softmax = param_softmax;
    return Status::OK();
}

template<>
Status SoftmaxHelper<X86, Precision::FP32>::Init(OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
template<>
Status SoftmaxHelper<X86, Precision::FP16>::Init(OpContext<X86>& ctx,
                                                           const std::vector<Tensor4dPtr<X86> >& ins,
                                                           std::vector<Tensor4dPtr<X86> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<>
Status SoftmaxHelper<X86, Precision::INT8>::Init(OpContext<X86>& ctx,
                                                           const std::vector<Tensor4dPtr<X86> >& ins,
                                                           std::vector<Tensor4dPtr<X86> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SoftmaxHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype>> &ins,
                           std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, STATIC, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SoftmaxHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                                 std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_softmax.compute_output_shape(ins, outs, _param_softmax));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SOFTMAX(NV, Precision::FP32);
template class SoftmaxHelper<NV, Precision::FP32>;
template class SoftmaxHelper<NV, Precision::FP16>;
template class SoftmaxHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_SOFTMAX(X86, Precision::FP32);
template class SoftmaxHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SOFTMAX(ARM, Precision::FP32);
template <>
Status SoftmaxHelper<ARM, Precision::FP32>::Init(OpContext<ARM> &ctx, \
    const std::vector<Tensor4dPtr<ARM> >& ins, \
    std::vector<Tensor4dPtr<ARM> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, ARM, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_SOFTMAX(AMD, Precision::FP32);
template <>
Status SoftmaxHelper<AMD, Precision::FP32>::Init(OpContext<AMD> &ctx, \
    const std::vector<Tensor4dPtr<AMD> >& ins, \
    std::vector<Tensor4dPtr<AMD> >& outs) {
    SABER_CHECK(_funcs_softmax.init(ins, outs, _param_softmax, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Softmax, SoftmaxHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Softmax)
.Doc("Softmax operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("softmax")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("softmax")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("softmax")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("softmax")
#endif
.num_in(1)
.num_out(1)
.Args<int>("axis", " axis ");

} /* namespace ops */

} /* namespace anakin */


