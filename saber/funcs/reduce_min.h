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

#ifndef ANAKIN_SABER_FUNCS_REDUCE_MIN_H
#define ANAKIN_SABER_FUNCS_REDUCE_MIN_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_reduce_min.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_reduce_min.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_reduce_min.h"
#endif

#ifdef USE_AMD
#endif

#ifdef USE_ARM_PLACE
#endif

#ifdef USE_BM
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class ReduceMin : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        ReduceMinParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            ReduceMinParam>::BaseFunc;

    ReduceMin() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef ReduceMinParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        
        Shape input_shape = input[0]->valid_shape();
        int input_dim = input_shape.size();
        // int real_dim = 0;
        // //Count with the real_dim that wanted to be reduced.
        // for (int i = 0; i < input_dim; ++i) {
        //     if (input_shape[i] != 1) {
        //         ++real_dim;
        //     }
        // }
        LOG(INFO) <<"input.valid.size:"<<input[0]->valid_size();
        Shape output_shape(input[0]->valid_shape());
        int reduce_dim = param.reduce_dim.size();
        //The dim we want to reduce is not empty.
        if (reduce_dim != 0) {
            //Check valid reduce dim.
            CHECK_LT(reduce_dim, input_dim) << "[reduce_min]reduce_dim's size must less than input's!!!";
            int tmp_dim;
            for (int i = 0; i < reduce_dim; i++) {
                if (param.reduce_dim[i] < 0) {
                    tmp_dim = param.reduce_dim[i] + input_dim;
                    CHECK_GE(tmp_dim, 0) << "[reduce_min] invalid reduce_dim!!!";
                    CHECK_LT(tmp_dim, input_dim) << "[reduce_min]invalid reduce_dim!!!";
                    output_shape[tmp_dim] = 1; //The dimention tmp_dim is to reduce dimention.
                }else {
                    CHECK_LT(param.reduce_dim[i], input_dim) << "[reduce_min]invalid reduce_dim!!!";
                    output_shape[param.reduce_dim[i]] = 1;
                }
                //output_shape[param.reduce_dim[i]] = 1;
            }
        }else {
            //Default to reduce all dimensions to a single value.
            output_shape = Shape({1, 1, 1, 1}); 
        }
        if (!param.keep_dim) {
            int size = output_shape.count();
            output_shape = Shape({size, 1, 1, 1});
        }

        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderReduceMin <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberReduceMin <TargetType, OpDtype>);
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
