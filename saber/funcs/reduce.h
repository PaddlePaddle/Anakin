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

#ifndef ANAKIN_SABER_FUNCS_REDUCE_H
#define ANAKIN_SABER_FUNCS_REDUCE_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_reduce.h"

#ifdef USE_CUDA
#include "saber/funcs/impl/cuda/saber_reduce.h"
#include "saber/funcs/impl/cuda/vender_reduce.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_reduce.h"
#endif

namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class Reduce : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        ReduceParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            ReduceParam>::BaseFunc;

    Reduce() = default;

    virtual SaberStatus compute_output_shape(
            const std::vector<Tensor<TargetType>*>& input,
            std::vector<Tensor<TargetType>*> &output,
            ReduceParam<TargetType> &param) override {
        Shape input_shape = input[0]->valid_shape();
        int input_dim = input_shape.size();
//        LOG(INFO) <<"input.valid.size:"<<input[0]->valid_size();

        int reduce_dim = param.reduce_dim.size();
        //The dim we want to reduce is not empty.
        if (param.reduce_all) {
            // CHECK IF reduce dim size is legal
            // I hope parser has handle this for saber,
            // if not, saber will re-write reduce_dim
            if (param.reduce_dim.size() != input_dim) {
                param.reduce_dim.clear();
                for (int i = 0; i < input_dim; ++i) {
                    param.reduce_dim.push_back(i);
                }
            }
            // check keep dim ?
            std::vector<int> temp_shape(input_dim, 1);
            Shape out_shape(temp_shape);
            return output[0]->set_shape(out_shape);
        } else  {
            //Check valid reduce dim.
            Shape output_shape(input[0]->valid_shape());
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
            return output[0]->set_shape(output_shape);
        }
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case VENDER_IMPL:
                this->_impl.push_back(new VenderReduce <TargetType, OpDtype>);
                return SaberSuccess;

            case SABER_IMPL:
                this->_impl.push_back(new SaberReduce <TargetType, OpDtype>);
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
