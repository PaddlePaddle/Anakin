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

#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_RESIZE_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_RESIZE_H

#include "saber/lite/funcs/op_base.h"

#ifdef USE_ARM_PLACE

namespace anakin{

    namespace saber{

        namespace lite{

//template <typename Dtype>
            class SaberResize : public OpBase {
            public:

                SaberResize() = default;

                SaberResize(const ParamBase* param);

                virtual SaberStatus load_param(const ParamBase* param) override;

                virtual SaberStatus load_param(std::istream& stream, const float* weights) override;

                ~SaberResize();


                virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                                         std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

                virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                         std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override;

                virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

            private:
                const ResizeParam* _param;
                ////coordinate buffer
                Tensor<CPU, AK_FLOAT> _coor_buf; 
                //rows buffer
                std::vector<Tensor<CPU, AK_FLOAT>> _rows_buf;
                float _width_scale{0.0f};
                float _height_scale{0.0f};
            };

        } //namespace lite

    } //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_RESIZE_H
