/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *     
*/

#include "framework/operators/slice.h"

namespace anakin {

namespace ops {

#define INSTANCE_SLICE(Ttype, Ptype) \
template<> \
void Slice<Ttype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SliceHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SliceHelper<Ttype, Ptype>*>(this->_helper)->_param_slice; \
    impl->_funcs_slice(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status SliceHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Slice op parameter.";
    auto slice_dim = GET_PARAMETER(int, slice_dim);
    _slice_point = GET_PARAMETER(PTuple<int>, slice_point);
    _axis = GET_PARAMETER(int, axis);
    DLOG(INFO) << " slice_dim " << slice_dim;
    DLOG(INFO) << " slice_point size(" << _slice_point.size() << ").";

    for (auto item : _slice_point.vector()) {
        DLOG(INFO) << "  |-- " << item;
    }

    DLOG(INFO) << " axis " << _axis;

    SliceParam<Ttype> param_slice(_axis, _slice_point.vector());
    _param_slice = param_slice;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SliceHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_slice.init(ins, outs, _param_slice, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SliceHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    if (_slice_point.size() + 1 != outs.size()) {
        if (_slice_point.size() == 1) {
            for (int i = 0; i < outs.size() - 2; i++) {
                _slice_point.push_back(_slice_point[0] + _slice_point[_slice_point.size() - 1]);
            }

            SliceParam<Ttype> param_slice(_axis, _slice_point.vector());
            _param_slice = param_slice;
        }
    }

    SABER_CHECK(_funcs_slice.compute_output_shape(ins, outs, _param_slice));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SLICE(NV, Precision::FP32);
template class SliceHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Slice, SliceHelper, NV, Precision::FP32);
template class SliceHelper<NV, Precision::FP16>;
template class SliceHelper<NV, Precision::INT8>;
#endif

#ifdef AMD_GPU
INSTANCE_SLICE(AMD, Precision::FP32);
template class SliceHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Slice, SliceHelper, AMD, Precision::FP32);
template class SliceHelper<AMD, Precision::FP16>;
template class SliceHelper<AMD, Precision::INT8>;
#endif

#if defined USE_X86_PLACE || defined(BUILD_LITE)
INSTANCE_SLICE(X86, Precision::FP32);
template class SliceHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Slice, SliceHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SLICE(ARM, Precision::FP32);
template class SliceHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Slice, SliceHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Slice)
.Doc("Slice operator")

#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("slice")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("slice")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("slice")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("slice")
#endif
.num_in(1)
.num_out(1)
.Args<int>("slice_dim", " slice dim at input ")
.Args<PTuple<int>>("slice_point", " slice point of op")
.Args<int>("axis", " axis of input to slice");

} /* namespace ops */

} /* namespace anakin */


