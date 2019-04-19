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

#include "framework/operators/slice_v2.h"

namespace anakin {

namespace ops {

#define INSTANCE_SLICE_V2(Ttype, Ptype) \
template<> \
void SliceV2<Ttype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SliceV2Helper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SliceV2Helper<Ttype, Ptype>*>(this->_helper)->_param_slice_v2; \
    impl->_funcs_slice_v2(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status SliceV2Helper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SliceV2 op parameter.";
    auto starts = GET_PARAMETER(PTuple<int>, starts);
    auto ends = GET_PARAMETER(PTuple<int>, ends);
    PTuple<int> axes;
    bool found_axes = CHECK_PARAMETER(axes);
    if (found_axes) {
        axes = GET_PARAMETER(PTuple<int>, axes);
    }
    DLOG(INFO) << " slice_v2 starts size(" << starts.size() << ").";
    DLOG(INFO) << " slice_v2 ends size(" << ends.size() << ").";
    DLOG(INFO) << " slice_v2 axes size(" << axes.size() << ").";
    std::vector<int> real_axes;
    if (axes.size() == 0) {
        real_axes.resize(starts.size());
        for (int i = 0; i < starts.size(); i++) {
            real_axes[i] = i;
        }
        SliceV2Param<Ttype> param_slice_v2(real_axes, starts.vector(), ends.vector());
        _param_slice_v2 = param_slice_v2;
    } else {
        int min_axes = axes.data()[0];
        int max_axes = axes.data()[axes.size() - 1];
        int axes_num = max_axes - min_axes + 1;
        std::vector<int> real_starts(axes_num, 0);
		std::vector<int> real_ends(axes_num, -1);
        std::vector<int> real_axes = axes.vector();
        if (axes_num == real_axes.size())  {
            real_starts = starts.vector();
            real_ends = ends.vector();
        } else {
            for (int i = 0; i < starts.size(); i++) {
                real_starts[axes.data()[i] - min_axes] = starts.data()[i];
                real_ends[axes.data()[i] - min_axes] = ends.data()[i];
            }
            real_axes.clear();
            for (int i = min_axes; i < max_axes; i++) {
                real_axes.push_back(i);
            }
        }
        SliceV2Param<Ttype> param_slice_v2(real_axes, real_starts, real_ends);
        _param_slice_v2 = param_slice_v2;
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SliceV2Helper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_slice_v2.init(ins, outs, _param_slice_v2, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SliceV2Helper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_slice_v2.compute_output_shape(ins, outs, _param_slice_v2));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SLICE_V2(NV, Precision::FP32);
template class SliceV2Helper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SliceV2, SliceV2Helper, NV, Precision::FP32);
template class SliceV2Helper<NV, Precision::FP16>;
template class SliceV2Helper<NV, Precision::INT8>;
#endif

#ifdef AMD_GPU
INSTANCE_SLICE_V2(AMD, Precision::FP32);
template class SliceV2Helper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SliceV2, SliceV2Helper, AMD, Precision::FP32);
template class SliceV2Helper<AMD, Precision::FP16>;
template class SliceV2Helper<AMD, Precision::INT8>;
#endif

#if defined USE_X86_PLACE || defined(BUILD_LITE)
INSTANCE_SLICE_V2(X86, Precision::FP32);
template class SliceV2Helper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SliceV2, SliceV2Helper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SLICE_V2(ARM, Precision::FP32);
template class SliceV2Helper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SliceV2, SliceV2Helper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(SliceV2)
.Doc("SliceV2 operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("slice_v2")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("slice_v2")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("slice_v2")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("slice_v2")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("starts", " slice_v2 start position ")
.Args<PTuple<int>>("ends", " slice_v2 end position ")
.Args<PTuple<int>>("axes", " slice_v2 axes position ");

} /* namespace ops */

} /* namespace anakin */


