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

#ifndef ANAKIN_SABER_FUNCS_SLICE_H
#define ANAKIN_SABER_FUNCS_SLICE_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_slice.h"
#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_slice.h"
#endif
#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_slice.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_slice.h"
#endif
#ifdef USE_ARM_PLACE
#include "saber/funcs/impl/arm/saber_slice.h"
#endif
namespace anakin{

namespace saber{

template <typename TargetType, DataType OpDtype>
class Slice : public BaseFunc<TargetType, OpDtype, ImplBase, SliceParam>
{
public:

    using BaseFunc<TargetType, OpDtype, ImplBase, SliceParam >::BaseFunc;
    Slice() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef SliceParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input, \
        Output_v &output, Param_t& param) override {

        SaberStatus status;
        //! input size is equal to 1
        Shape shape_in = input[0]->valid_shape();
        int top_size = output.size();
        int slice_points_size = param.slice_points.size();
        int axis_size = shape_in[param.axis];

        CHECK_EQ(top_size > 0 || slice_points_size > 0, true) << \
            "output shapes number is 0 and slice points size is 0";

        if (slice_points_size > 0) {
            CHECK_EQ(slice_points_size + 1, top_size) << "error params or ouput size";
            int prev = 0;
            Shape sh = shape_in;
            for (int i = 0; i < slice_points_size; ++i) {
                CHECK_GT(param.slice_points[i], prev) << " later should > prev";
                CHECK_LT(param.slice_points[i], axis_size) << "slice point exceed";
                sh[param.axis] = param.slice_points[i] - prev;
                output[i]->set_shape(sh);
                prev = param.slice_points[i];
                sh = shape_in;
            }
            CHECK_GT(axis_size - prev, 0) << "slice point exceed";
            sh[param.axis] = axis_size - prev;
            return output[slice_points_size]->set_shape(sh);
        } else {

            CHECK_EQ(axis_size % top_size, 0) << \
                "size in slice axis should divide exactly by top size";
            int step = axis_size / top_size;
            Shape sh = shape_in;
            sh[param.axis] = step;
            output[0]->set_shape(sh);
            param.slice_points.clear();
            for (int i = 1; i < top_size; ++i) {
                param.slice_points.push_back(i * step);
                status = output[i]->set_shape(sh);
                if (status != SaberSuccess) {
                    return status;
                }
            }
        }
        return SaberSuccess;
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) { 
            case VENDER_IMPL: 
                this->_impl.push_back(new VenderSlice <TargetType, OpDtype>); 
                return SaberSuccess; 
            case SABER_IMPL: 
                this->_impl.push_back(new SaberSlice <TargetType, OpDtype>); 
                return SaberSuccess; 
            default: 
                return SaberUnImplError;
        }        
    }

private:

    virtual void pick_best_static() override {
        //! slice only has saber implementation
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        //! slice only has saber implementation
        this->_best_impl = this->_impl[0];
    }

};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_SLICE_H
