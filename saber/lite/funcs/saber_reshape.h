/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_RESHAPE_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_RESHAPE_H

#include "saber/lite/funcs/op_base.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberReshape : public OpBase {
public:

    SaberReshape() = default;

    SaberReshape(const ParamBase* param) {
        _param = (ReshapeParam*)param;
        this->_flag_param = true;
    }

    virtual SaberStatus load_param(const ParamBase* param) override {
        _param = (ReshapeParam*)param;
        this->_flag_param = true;
        return SaberSuccess;
    }

    ~SaberReshape() {}


    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                              std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override {

        if (!this->_flag_param) {
            printf("load concat param first\n");
            return SaberNotInitialized;
        }

        Shape output_shape;
        output_shape.resize(_param->_shape_params.size());
        Shape input_shape = inputs[0]->valid_shape();
        int valid_size = inputs[0]->valid_size();
        int infer_axis = -1;
        int count_axis = 1;
        for (int i = 0; i < _param->_shape_params.size(); ++i) {
            if (_param->_shape_params[i] == 0){
                LCHECK_LT(i, input_shape.size(), "wrong parameters, exceed input dims");
                output_shape[i] = input_shape[i];
                count_axis *= input_shape[i];
            } else if (_param->_shape_params[i] > 0){
                output_shape[i] = _param->_shape_params[i];
                count_axis *= _param->_shape_params[i];
            } else {
                output_shape[i] = -1;
                infer_axis = i;
            }
        }

        if (infer_axis >= 0){
            output_shape[infer_axis] = valid_size / count_axis;
        }
        return outputs[0]->set_shape(output_shape);
    }

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                               std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override {

        if (!this->_flag_param) {
            printf("load concat param first\n");
            return SaberNotInitialized;
        }
        //outputs[0]->share_from(*inputs[0]);

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override {
        return SaberSuccess;
    }

private:
    const ReshapeParam* _param;
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_RESHAPE_H
