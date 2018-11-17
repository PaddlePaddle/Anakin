
#include "saber/core/context.h"
#include "saber/funcs/slice.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber_types.h"
#include <vector>

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

TEST(TestSaberFunc, test_func_slice){
#ifdef USE_CUDA
    LOG(INFO)<<"NV test......";
    //test 0
    TestSaberBase<NV, NVHX86, AK_FLOAT, Slice, SliceParam> testbase(1,4);
    int num_in = 4;
    int c_in = 9;
    int h_in = 12;
    int w_in = 12;
    int slice_axis = 1;
    std::vector<int> slice_points = {1,3,6};
    SliceParam<NV> param(slice_axis, slice_points);
    testbase.set_param(param);
    testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase.run_test(slice_cpu<float, NV, NVHX86>);
   
    //test1
    TestSaberBase<NV, NVHX86, AK_FLOAT, Slice, SliceParam> testbase1(1,4);
    num_in = 10;
    c_in = 3;
    h_in = 2;
    w_in = 3;
    slice_axis = 0;
    slice_points = {4,6,8};
    SliceParam<NV> param1(slice_axis, slice_points);
    testbase1.set_param(param1);
    testbase1.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase1.run_test(slice_cpu<float, NV, NVHX86>);
    //test2
    TestSaberBase<NV, NVHX86, AK_FLOAT, Slice, SliceParam> testbase2(1,2);
    num_in = 6;
    c_in = 4;
    h_in = 10;
    w_in = 2;
    slice_axis = 2;
    slice_points = {5};
    SliceParam<NV> param2(slice_axis, slice_points);
    testbase2.set_param(param2);
    testbase2.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase2.run_test(slice_cpu<float, NV, NVHX86>);
    //test3
    TestSaberBase<NV, NVHX86, AK_FLOAT, Slice, SliceParam> testbase3(1,3);
    num_in = 10;
    c_in = 11;
    h_in = 1;
    w_in = 11;
    slice_axis = 3;
    slice_points = {1,9};
    SliceParam<NV> param3(slice_axis, slice_points);
    testbase3.set_param(param3);
    testbase3.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
    testbase3.run_test(slice_cpu<float, NV, NVHX86>);
#endif

#ifdef USE_X86_PLACE
    LOG(INFO)<<"x86 test......";
    do
    {
        //test 0
        TestSaberBase<X86, X86, AK_FLOAT, Slice, SliceParam> testbase(1,4);
        int num_in = 4;
        int c_in = 9;
        int h_in = 12;
        int w_in = 12;
        int slice_axis = 1;
        std::vector<int> slice_points = {1,3,6};
        SliceParam<X86> param(slice_axis, slice_points);
        testbase.set_param(param);
        testbase.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase.run_test(slice_cpu<float, X86, X86>);

        //test1
        TestSaberBase<X86, X86, AK_FLOAT, Slice, SliceParam> testbase1(1,4);
        num_in = 10;
        c_in = 3;
        h_in = 2;
        w_in = 3;
        slice_axis = 0;
        slice_points = {4,6,8};
        SliceParam<X86> param1(slice_axis, slice_points);
        testbase1.set_param(param1);
        testbase1.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase1.run_test(slice_cpu<float, X86, X86>);

        //test2
        TestSaberBase<X86, X86, AK_FLOAT, Slice, SliceParam> testbase2(1,2);
        num_in = 6;
        c_in = 4;
        h_in = 10;
        w_in = 2;
        slice_axis = 2;
        slice_points = {5};
        SliceParam<X86> param2(slice_axis, slice_points);
        testbase2.set_param(param2);
        testbase2.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase2.run_test(slice_cpu<float, X86, X86>);
        //test3
        TestSaberBase<X86, X86, AK_FLOAT, Slice, SliceParam> testbase3(1,3);
        num_in = 10;
        c_in = 11;
        h_in = 1;
        w_in = 11;
        slice_axis = 3;
        slice_points = {1,9};
        SliceParam<X86> param3(slice_axis, slice_points);
        testbase3.set_param(param3);
        testbase3.set_input_shape(Shape({num_in, c_in, h_in, w_in}));
        testbase3.run_test(slice_cpu<float, X86, X86>);

    }while(0);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

