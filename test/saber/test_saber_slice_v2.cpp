#include "saber/core/context.h"
#include "saber/funcs/slice_v2.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;


template <typename dtype,typename TargetType_D,typename TargetType_H>
void slice_v2_cpu(const std::vector<Tensor<TargetType_H>*>& inputs,
                  std::vector<Tensor<TargetType_H>*>& outputs,\
                  SliceV2Param<TargetType_D>& param){

    auto starts = param.starts;
    auto ends = param.ends;
    auto axes = param.axes;
    CHECK_EQ(axes.size(), starts.size()) << "the size of axes and starts are not equal ";
    CHECK_EQ(ends.size(), starts.size()) << "the size of starts and ends are not valid";
    Shape shape_in = inputs[0]->valid_shape();
    Shape out_shape = shape_in;
    std::vector<int> valid_starts;
    std::vector<int> valid_ends;
    valid_starts.resize(starts.size());
    valid_ends.resize(ends.size());
    for (int i = 0; i < starts.size(); i++) {
        int dim_value = shape_in[axes[i]];
        int start = starts[i] < 0 ? starts[i] + dim_value : starts[i];
        int end = ends[i] < 0 ? ends[i] + dim_value : ends[i];
        start = std::max(start, 0);
        start = std::min(start, dim_value);
        end = std::max(end, 0);
        end = std::min(end, dim_value);
        out_shape[axes[i]] = end - start;
        valid_starts[i] = start;
        valid_ends[i] = end;
    }
    CHECK_EQ(outputs.size(), 1) << "SliceV2 only support one output";
    const dtype* in_data = (const dtype*)inputs[0]->data();
    dtype* out_data = (dtype*)outputs[0]->mutable_data();
    auto out_stride = outputs[0]->get_stride();
    auto in_stride = inputs[0]->get_stride();
    int inner = inputs[0]->count_valid(param.axes.back() + 1, outputs[0]->dims());
    int out_outer_stride = outputs[0]->count_valid(param.axes[0], outputs[0]->dims());
    int in_outer_stride = inputs[0]->count_valid(param.axes[0], inputs[0]->dims());
    int count = outputs[0]->valid_size();

    for (int i = 0; i < count; i++) {
        int out_id = i / out_outer_stride;
        int inner_id = i % inner;
        int new_i = i / inner;
        int in_offset = inner_id + out_id * in_outer_stride;
        for (int k = valid_starts.size() - 1; k >= 0; k--) {
            int cur_id = new_i % out_shape[axes[k]];
            in_offset += (cur_id + valid_starts[k]) * in_stride[axes[k]];
            new_i /= out_shape[axes[k]];
        }
        out_data[i] = in_data[in_offset];
    }

}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {
    Shape input_shape({2, 5, 2, 2}, Layout_NCHW);
    std::vector<int> starts_0 = {1, 0};
    std::vector<int> ends_0 = {3, 1};
    std::vector<int> axes_0 = {1, 2};
    std::vector<int> starts_1 = {0, 1, 0, 1};
    std::vector<int> ends_1 = {1, 3, 1, 2};
    std::vector<int> axes_1 = {0, 1, 2, 3};
    std::vector<int> starts_2 = {1};
    std::vector<int> ends_2 = {3};
    std::vector<int> axes_2 = {1};

    TestSaberBase<TargetType_D, TargetType_H, Dtype, SliceV2, SliceV2Param> testbase(1, 1);
    for (auto i : {0, 1, 2}) {
        std::vector<int> axes;
        std::vector<int> starts;
        std::vector<int> ends;
        if (i == 0) {
            axes = axes_0;
            starts = starts_0;
            ends = ends_0;
        } else if (i == 1) {
            axes = axes_1;
            starts = starts_1;
            ends = ends_1;
        } else if (i == 2) {
            axes = axes_2;
            starts = starts_2;
            ends = ends_2;
        } else {
            LOG(FATAL) << "no other param";
        }
        SliceV2Param<TargetType_D> param(axes, starts, ends);
	    testbase.set_param(param);//set param
	    testbase.set_input_shape(input_shape);
	    testbase.run_test(slice_v2_cpu<float, TargetType_D, TargetType_H>, 0.0001, true, false);
    }
}

TEST(TestSaberFunc, test_func_slice_v2) {

#ifdef USE_CUDA
    //Init the test_base
    test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    test_model<AK_FLOAT, X86, X86>();
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

