/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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

#include <vector>

#include "saber/core/context.h"
#include "saber/funcs/slice.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber_types.h"

using namespace anakin::saber;


template <typename dtype,typename TargetType_D,typename TargetType_H>
void slice_cpu(const std::vector<Tensor<TargetType_H>*>& input,std::vector<Tensor<TargetType_H>*>& output,\
                SliceParam<TargetType_D>& param){


    int slice_num = input[0]->count_valid(0, param.axis);
    int slice_size = input[0]->count_valid(param.axis + 1, input[0]->dims());
    Shape shape_in = input[0]->valid_shape();
    int out_size = output.size();
    const dtype* in = (const dtype*)input[0]->data();
    const int in_slice_axis_size = shape_in[param.axis];
    int offset_slice_axis = 0;
    for (int i = 0; i < out_size; ++i){
        dtype* out = (dtype*)output[i]->mutable_data();
        const int out_slice_axis_size = output[i]->valid_shape()[param.axis];
        const int out_slice_size = out_slice_axis_size * slice_size;
        const int slice_count = out_slice_size * slice_num;
        for (int j = 0; j < slice_count; ++j){
            const int _num_slice = j / out_slice_size;
            const int _slice_index = j % out_slice_size;
            const int in_index = _slice_index + (_num_slice * in_slice_axis_size + offset_slice_axis) * slice_size;
            out[j] = in[in_index];
        }
        offset_slice_axis += out_slice_axis_size;
    }

}

struct TestCase {
    int n;
    int c;
    int h;
    int w;
    int slice_axis;
    std::vector<int> slice_points;
};

template <typename dtype,typename TargetD,typename TargetH>
void test_slice(
        const TestCase& t
        , double succ_ratio = 0.00001) {
    TestSaberBase<TargetD, TargetH, AK_FLOAT, Slice, SliceParam> testbase(
        1, t.slice_points.size() + 1);

    SliceParam<TargetD> param(t.slice_axis, t.slice_points);
    testbase.set_param(param);
    testbase.set_input_shape(Shape({t.n, t.c, t.h, t.w}));
    testbase.run_test(slice_cpu<dtype, TargetD, TargetH>, succ_ratio);
}

TEST(TestSaberFunc, test_func_slice) {
    std::vector<TestCase> test_cases{
        {3, 9, 12, 12, 1, {1, 3, 6}},
        {10, 3, 2, 3, 0, {4, 6, 8}},
        {6, 4, 19, 2, 2, {5}},
        {10, 11, 1, 11, 3, {1, 9}},
    };

#ifdef USE_CUDA
    LOG(INFO)<<"NV test......";
    for (const auto& t : test_cases) {
        test_slice<float, NV, NVHX86>(t);
    }
#endif // ifdef USE_CUDA

#ifdef USE_X86_PLACE
    LOG(INFO)<<"x86 test......";
    for (const auto& t : test_cases) {
        test_slice<float, X86, X86>(t);
    }
#endif // ifdef USE_X86_PLACE

#ifdef USE_ARM_PLACE
    LOG(INFO)<<"ARM test......";
    for (const auto& t : test_cases) {
        test_slice<float, ARM, ARM>(t);
    }
#endif // ifdef USE_ARM_PLACE

#ifdef USE_MLU
    LOG(INFO)<<"MLU test......";
    for (const auto& t : test_cases) {
        test_slice<float, MLU, MLUHX86>(t, 0.02);
    }
#endif // ifdef USE_MLU
}

int main(int argc, const char** argv) {
    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}
