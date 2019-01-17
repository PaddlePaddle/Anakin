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
#include "framework/operators/split.h"

namespace anakin {

namespace ops {
#define INSTANCE_SPLIT(Ttype, Ptype) \
template<> \
void Split<Ttype, Ptype>::operator()( \
        OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
        std::vector<Tensor4dPtr<Ttype> >& outs) {}

template<typename Ttype, Precision Ptype>
Status SplitHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Split op parameter.";
    split_num = GET_PARAMETER(int, split_num);
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SplitHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype>> &ins,
                         std::vector<Tensor4dPtr<Ttype>> &outs) {
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SplitHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                               std::vector<Tensor4dPtr<Ttype>> &outs) {
    for (int i = 0; i < split_num; i++) {
        outs[i]->set_shape_without_layout(ins[0]->valid_shape());
        outs[i]->set_seq_offset(ins[0]->get_seq_offset());
    }
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SPLIT(NV, Precision::FP32);
template class SplitHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Split, SplitHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SPLIT(ARM, Precision::FP32);
template class SplitHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Split, SplitHelper, ARM, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_SPLIT(X86, Precision::FP32);
template class SplitHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Split, SplitHelper, X86, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_SPLIT(AMD, Precision::FP32);
template class SplitHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Split, SplitHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Split)
.Doc("Split operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("split")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("split")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("split")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("split")
#endif
.num_in(1)
.num_out(1)
.Args<int>("split_num", " split output number. ");


} /* namespace ops */

} /* namespace anakin */


