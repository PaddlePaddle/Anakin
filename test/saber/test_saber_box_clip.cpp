#include "core/context.h"
#include "saber/funcs/box_clip.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test/saber/test_saber_base.h"
#include <vector>

using namespace anakin::saber;

template <typename dtype, typename TargetType_D, typename TargetType_H>
void box_clip_basic(const std::vector<Tensor<TargetType_H>*>& inputs,
                    std::vector<Tensor<TargetType_H>*>& outputs, BoxClipParam<TargetType_D>& param) {
    static constexpr int im_info_size = 3;
    static constexpr int box_info_size = 4;
    auto seq_offset = inputs[1]->get_seq_offset();
    CHECK_EQ(inputs.size(), 2) << "need two input";
    CHECK_EQ(seq_offset.size(), 1) << "need offset to cal batch";
    CHECK_GT(seq_offset[0].size(), 1) << "need offset to cal batch";
    auto offset = seq_offset[0];
    auto img = inputs[1];
    auto im_info = inputs[0];
    const float* im_info_ptr = static_cast<const float*>(im_info->data());
    const float* box_ptr_in = static_cast<const float*>(img->data());
    float* box_ptr_out = static_cast<float*>(outputs[0]->data());
    int batch_size = offset.size() - 1;
    CHECK_EQ(batch_size * im_info_size, im_info->valid_size()) << "im_info should be valid";

    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        const float img_h = im_info_ptr[batch_id * im_info_size + 0];
        const float img_w = im_info_ptr[batch_id * im_info_size + 1];
        const float scale = im_info_ptr[batch_id * im_info_size + 2];
        const float img_h_scale = round(img_h / scale) - 1;
        const float img_w_scale = round(img_w / scale) - 1;
        const int start_in_batch = offset[batch_id];
        const int end_in_batch = offset[batch_id + 1];

        for (int im_id = start_in_batch; im_id < end_in_batch; im_id++) {
            const float* batch_box_ptr_in = &box_ptr_in[im_id * box_info_size];
            float* batch_box_ptr_out = &box_ptr_out[im_id * box_info_size];
            batch_box_ptr_out[0] = std::max(std::min(batch_box_ptr_in[0], img_w_scale), 0.f);
            batch_box_ptr_out[1] = std::max(std::min(batch_box_ptr_in[1], img_h_scale), 0.f);
            batch_box_ptr_out[2] = std::max(std::min(batch_box_ptr_in[2], img_w_scale), 0.f);
            batch_box_ptr_out[3] = std::max(std::min(batch_box_ptr_in[3], img_h_scale), 0.f);
        }
    }
}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_model() {

    int batch = 2;
    int box_per_batch = 2;
    int num = box_per_batch * batch;
    int channel = 4;
    int height = 1;
    int width = 1;

    TestSaberBase<TargetType_D, TargetType_H, Dtype, BoxClip, BoxClipParam> testbase(2, 1);

    BoxClipParam<TargetType_D> param;

    Shape input_shape({num, channel, height, width}, Layout_NCHW);
    Shape im_info_shape({batch, 3, 1, 1}, Layout_NCHW);
    Tensor<TargetType_D> input_box_host(input_shape);
    Tensor<TargetType_D> im_info_host(im_info_shape);
    fill_tensor_rand(input_box_host, 0, 100);
    fill_tensor_rand(im_info_host, 0, 100);
    std::vector<std::vector<int>> seq_offset({{0}});

    for (int i = 1; i <= batch; i++) {
        seq_offset[0].push_back(seq_offset[0][i - 1] + box_per_batch);
    }

    input_box_host.set_seq_offset(seq_offset);
    std::vector<Tensor<TargetType_D>*> input_vec;
    input_vec.push_back(&im_info_host);
    input_vec.push_back(&input_box_host);
    testbase.set_param(param);//set param
    testbase.add_custom_input(input_vec);
    testbase.run_test(box_clip_basic<float, TargetType_D, TargetType_H>);//run test


}

TEST(TestSaberFunc, test_func_axpy) {

#ifdef USE_CUDA
    //Init the test_base
//    test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    test_model<AK_FLOAT, X86, X86>();
#endif
}


int main(int argc, const char** argv) {

    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}

