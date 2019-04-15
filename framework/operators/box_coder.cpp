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
#include "framework/operators/box_coder.h"

namespace anakin {

namespace ops {

/// TODO ... specialization other type of operator
#define INSTANCE_AXPY(Ttype, Ptype) \
template<> \
void BoxCoder<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<BoxCoderHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_box_coder; \
    impl->_funcs_box_coder(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
BoxCoderHelper<Ttype, Ptype>::~BoxCoderHelper() {
}

template<typename Ttype, Precision Ptype>
Status BoxCoderHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing BoxCoder op parameter.";
    auto axis = GET_PARAMETER(int, axis);
    auto box_normalized = GET_PARAMETER(bool, box_normalized);
    Tensor<Ttype>* variance = nullptr;

    if (FIND_PARAMETER(variance)) {
        variance = &((GET_PARAMETER(PBlock<Ttype>, variance)).d_tensor());
    }

    saber::BoxCoderParam<Ttype> box_coder_param(variance, box_normalized, axis);
    _param_box_coder = box_coder_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status BoxCoderHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_box_coder.init(ins, outs, _param_box_coder, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status BoxCoderHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_box_coder.compute_output_shape(ins, outs, _param_box_coder));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_AXPY(NV, Precision::FP32);
template class BoxCoderHelper<NV, Precision::FP32>;
template class BoxCoderHelper<NV, Precision::FP16>;
template class BoxCoderHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(BoxCoder, BoxCoderHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_AXPY(AMD, Precision::FP32);
template class BoxCoderHelper<AMD, Precision::FP32>;
template class BoxCoderHelper<AMD, Precision::FP16>;
template class BoxCoderHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(BoxCoder, BoxCoderHelper, AMD, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_AXPY(X86, Precision::FP32);
template class BoxCoderHelper<X86, Precision::FP32>;
template class BoxCoderHelper<X86, Precision::FP16>;
template class BoxCoderHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(BoxCoder, BoxCoderHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE

#ifdef ANAKIN_TYPE_FP32
INSTANCE_AXPY(ARM, Precision::FP32);
template class BoxCoderHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BoxCoder, BoxCoderHelper, ARM, Precision::FP32);
#endif

#ifdef ANAKIN_TYPE_FP16
template class BoxCoderHelper<ARM, Precision::FP16>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class BoxCoderHelper<ARM, Precision::INT8>;
#endif

#endif//arm

//! register op
ANAKIN_REGISTER_OP(BoxCoder)
.Doc("BoxCoder operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("box_coder")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("box_coder")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("box_coder")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("box_coder")
#endif
.num_in(3)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


