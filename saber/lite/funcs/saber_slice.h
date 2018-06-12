/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_SLICE_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_SLICE_H
#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{
template <typename Dtype>
class SaberSlice {
public:

    SaberSlice() {
        _slice_num = 4;
        _slice_size = 0;
    }
    ~SaberSlice() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     SliceParam<Tensor<Dtype>> &param) {
        SaberStatus status;
        //! input size is equal to 1
        Shape shape_in = inputs[0]->valid_shape();
        int top_size = outputs.size();
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
                outputs[i]->set_shape(sh);
                prev = param.slice_points[i];
                sh = shape_in;
            }
            CHECK_GT(axis_size - prev, 0) << "slice point exceed";
            sh[param.axis] = axis_size - prev;
            return outputs[slice_points_size]->set_shape(sh);
        } else {

            CHECK_EQ(axis_size % top_size, 0) << \
                "size in slice axis should divide exactly by top size";
            int step = axis_size / top_size;
            Shape sh = shape_in;
            sh[param.axis] = step;
            outputs[0]->set_shape(sh);
            for (int i = 1; i < top_size; ++i) {
                param.slice_points[i - 1] = i * step;
                status = outputs[i]->set_shape(sh);
                if (status != SaberSuccess) {
                    return status;
                }
            }
        }
        return SaberSuccess;
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs,
                             std::vector<Tensor<Dtype>*>& outputs,
                             SliceParam<Tensor<Dtype>> &param, Context &ctx) {
        // get context
        return create(inputs, outputs, param ,ctx);
    }

    SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs,
                               std::vector<Tensor<Dtype>*>& outputs,
                               SliceParam<Tensor<Dtype>> &param, Context &ctx) {
        _ctx = ctx;
        _slice_num = inputs[0]->count_valid(0, param.axis);
        _slice_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
       return SaberSuccess;
    }

    SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs,
                                 std::vector<Tensor<Dtype>*>& outputs,
                                 SliceParam<Tensor<Dtype>> &param);

private:
    Context _ctx;
    int _slice_num;
    int _slice_size;
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_SLICE_H
