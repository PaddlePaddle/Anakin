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

#ifndef ANAKIN_SABER_FUNCS_SLICE_V2_H
#define ANAKIN_SABER_FUNCS_SLICE_V2_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_slice_v2.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_slice_v2.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_slice_v2.h"
#endif

#ifdef AMD_GPU
//#include "saber/funcs/impl/amd/include/saber_slice_v2.h"
#endif

#ifdef USE_ARM_PLACE
//#include "saber/funcs/impl/arm/saber_slice_v2.h"
#endif

#ifdef USE_BM_PLACE 
//#include "saber/funcs/impl/bm/vender_slice_v2.h"
#endif


namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class SliceV2 : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        SliceV2Param> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            SliceV2Param>::BaseFunc;

    SliceV2() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef SliceV2Param<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {

        Shape output_shape = input[0]->valid_shape();
        Shape in_shape = input[0]->valid_shape();
        auto starts = param.starts;
        auto ends = param.ends;
        auto axes = param.axes;
        CHECK_EQ(axes.size(), starts.size()) << "the size of axes and starts are not equal ";
        CHECK_EQ(ends.size(), starts.size()) << "the size of starts and ends are not valid";
        for (int i = 0; i < starts.size(); i++) {
            int dim_value = in_shape[axes[i]];
            int start = starts[i] < 0 ? starts[i] + dim_value : starts[i];
            int end = ends[i] < 0 ? ends[i] + dim_value : ends[i];
            start = std::max(start, 0);
            start = std::min(start, dim_value);
            end = std::max(end, 0);
            end = std::min(end, dim_value);
            output_shape[axes[i]] = end - start;
        }
        if (axes[0] != 0) {
            output[0]->set_seq_offset(input[0]->get_seq_offset());
        }
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                //this->_impl.push_back(new VenderSliceV2 <TargetType,
                this->_impl.push_back(new VenderSliceV2 <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberSliceV2 <TargetType, OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};



} // namespace saber
} // namespace anakin

#endif
