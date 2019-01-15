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
#include "framework/operators/axpy.h"

namespace anakin {

namespace ops {

/// TODO ... specialization other type of operator
#define INSTANCE_AXPY(Ttype, Ptype) \
template<> \
void Axpy<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<AxpyHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_axpy; \
    impl->_funcs_axpy(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
AxpyHelper<Ttype, Ptype>::~AxpyHelper() {
}

template<typename Ttype, Precision Ptype>
Status AxpyHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Axpy op parameter.";

    saber::AxpyParam<Ttype> axpy_param;
    _param_axpy = axpy_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status AxpyHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_axpy.init(ins, outs, _param_axpy, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status AxpyHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_axpy.compute_output_shape(ins, outs, _param_axpy));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_AXPY(NV, Precision::FP32);
template class AxpyHelper<NV, Precision::FP32>;
template class AxpyHelper<NV, Precision::FP16>;
template class AxpyHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Axpy, AxpyHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_AXPY(AMD, Precision::FP32);
template class AxpyHelper<AMD, Precision::FP32>;
template class AxpyHelper<AMD, Precision::FP16>;
template class AxpyHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Axpy, AxpyHelper, AMD, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_AXPY(X86, Precision::FP32);
template class AxpyHelper<X86, Precision::FP32>;
template class AxpyHelper<X86, Precision::FP16>;
template class AxpyHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Axpy, AxpyHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE

#ifdef ANAKIN_TYPE_FP32
INSTANCE_AXPY(ARM, Precision::FP32);
template class AxpyHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Axpy, AxpyHelper, ARM, Precision::FP32);
#endif

#ifdef ANAKIN_TYPE_FP16
template class AxpyHelper<ARM, Precision::FP16>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class AxpyHelper<ARM, Precision::INT8>;
#endif

#endif//arm

//! register op
ANAKIN_REGISTER_OP(Axpy)
.Doc("Axpy operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("axpy")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("axpy")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("axpy")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("axpy")
#endif
.num_in(3)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


