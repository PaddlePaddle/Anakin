
#include "saber/core/context.h"
#include "saber/funcs/sequence_conv.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>



using namespace anakin::saber;


template <typename Dtype>
static void im2col_2d_ocf(const Dtype* in, int stride, int pad_up, int pad_down, int kernel_size,
                          Dtype* out, int seq_length, int hidden_size) {
    for (int out_row = 0; out_row < seq_length; ++out_row) {
        for (int col = 0; col < kernel_size; ++col) {
            int index = out_row + col - pad_up;
            int out_index = (out_row * kernel_size + col) * hidden_size;

            for (int hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
                if (index < 0 || index >= seq_length) {
                    out[out_index + hidden_index] = 0;
                } else {
                    out[out_index + hidden_index] = in[index * hidden_size * stride + hidden_index];
                }
            }
        }
    }
}

static void gemm(int m, int n, int k, float alpha, const float* A, const float* B, float beta,
                 float* C) {
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            float sum = 0.0f;

            for (int inner = 0; inner < k; ++inner) {
                sum += A[j * k + inner] * B[inner * n + i];
            }

            C[j * n + i] = alpha * sum + beta * C[j * n + i];
        }
    }
}

template <typename dtype, typename TargetType_D, typename TargetType_H>
void sequence_conv_cpu(const std::vector<Tensor<TargetType_H>*>& inputs,
                       std::vector<Tensor<TargetType_H>*>& outputs, \
                       SequenceConvParam<TargetType_D>& param) {

    Tensor<X86> _temp_im2col_tensor;
    Tensor<X86> temp_filter_tensor;
    int _hidden_size = param.filter_tensor->height() / param.context_length;
    int _feature_size = param.filter_tensor->width();
    int _up_pad = std::max(0, -param.context_start);
    int _down_pad = std::max(0, param.context_start + param.context_length - 1);
    int _hidden_kernel_size = _hidden_size * param.context_length;

    Tensor<TargetType_H>* in_data = inputs[0];
    Tensor<TargetType_H>* out_data = outputs[0];
    std::vector<std::vector<int>> voffset = in_data->get_seq_offset();
    out_data->set_seq_offset(voffset);
    std::vector<int> offset = voffset[0];
    int word_num = offset[offset.size() - 1];

    Shape sh_im({1, 1, word_num, param.filter_tensor->height()});
    _temp_im2col_tensor.re_alloc(sh_im, AK_FLOAT);
    const float* in = (const float*)in_data->data();
    float* out = (float*)out_data->mutable_data();
    float* im2col = (float*)_temp_im2col_tensor.mutable_data();
    temp_filter_tensor.set_shape(param.filter_tensor->valid_shape());
    temp_filter_tensor.copy_from(*(param.filter_tensor));

    for (int i = 0; i < offset.size() - 1; ++i) {
        int start = offset[i];
        int seq_length = offset[i + 1] - offset[i];
        im2col_2d_ocf(in + _hidden_size * start, param.context_stride, _up_pad, _down_pad, \
                      param.context_length, im2col + _hidden_kernel_size * start, seq_length, _hidden_size);
    }

    gemm(word_num, _feature_size, _hidden_kernel_size, 1.f, (const float*)im2col,
         (const float*)temp_filter_tensor.data(), 0.f, out);

    out_data->set_seq_offset(voffset);
}

static void get_seq_offset(int num, std::vector<int>& seq_offset) {
    int seg_num = 4;
    seq_offset.resize(seg_num + 1);

    for (int i = 0; i < seg_num; ++i) {
        seq_offset[i] = i * num / seg_num;
    }

    seq_offset[seg_num] = num;
}

TEST(TestSaberFunc, test_func_saber_sequenconv) {

#ifdef USE_CUDA
    /*
    LOG(INFO) << "NV test......";
    typedef Tensor<NV> TensorD;
    //Init the test_base
    TestSaberBase<NV, NVHX86, AK_FLOAT, SequenceConv, SequenceConvParam> testbase;

    for (auto num : {8, 16}) {
    for (auto hidden_size : {10, 20, 31}) {
    for (auto context_length : {2, 3, 7}) {
    for (auto feature_size : {4, 10, 64}) {
    for (auto pad_up : {-1, -2}) {
        LOG(INFO) << "num: " << num << ", hidden_size: " << hidden_size \
                  << ", context_length: " << context_length << ", feature_size: " << feature_size\
                  << ", pad_up: " << pad_up;
        TensorD filter_tensor;
        TensorD in_tensor;
        std::vector<TensorD*> input;
        Shape in_sh({num, hidden_size, 1, 1});
        Shape filter_sh({1, 1, hidden_size * context_length, feature_size});
        in_tensor.re_alloc(in_sh, AK_FLOAT);
        filter_tensor.re_alloc(filter_sh, AK_FLOAT);
        fill_tensor_rand(filter_tensor, -1.0f, 1.0f);
        fill_tensor_rand(in_tensor, -1.0f, 1.0f);
        std::vector<int> seq_offset;
        std::vector<std::vector<int>> vseq_offset;
        get_seq_offset(num, seq_offset);
        vseq_offset.push_back(seq_offset);
        in_tensor.set_seq_offset(vseq_offset);
        input.push_back(&in_tensor);
        SequenceConvParam<NV> param(&filter_tensor, context_length, pad_up);
        testbase.set_param(param);
        testbase.add_custom_input(input);
        testbase.run_test(sequence_conv_cpu<float, NV, NVHX86>);
    }
    }
    }
    }
    }

    LOG(INFO) << "NV end......";
*/
#endif

#ifdef USE_X86_PLACE
    LOG(INFO) << "x86 test......";

    do {
        typedef Tensor<X86> TensorD;
        //Init the test_base
        TestSaberBase<X86, X86, AK_FLOAT, SequenceConv, SequenceConvParam> testbase;

    for (auto num : {8, 16}) {
    for (auto hidden_size : {10, 20, 31}) {
    for (auto context_length : {2, 3, 7}) {
    for (auto feature_size : {4, 10, 64}) {
    for (auto pad_up : {-1, -2}) {
        LOG(INFO) << "num: " << num << ", hidden_size: " << hidden_size \
                  << ", context_length: " << context_length << ", feature_size: " << feature_size\
                  << ", pad_up: " << pad_up;
        TensorD filter_tensor;
        TensorD in_tensor;
        std::vector<TensorD*> input;
        Shape in_sh({num, hidden_size, 1, 1});
        Shape filter_sh({1, 1, hidden_size * context_length, feature_size});
        in_tensor.re_alloc(in_sh, AK_FLOAT);
        filter_tensor.re_alloc(filter_sh, AK_FLOAT);
        fill_tensor_rand(filter_tensor, -1.0f, 1.0f);
        fill_tensor_rand(in_tensor, -1.0f, 1.0f);
        std::vector<int> seq_offset;
        std::vector<std::vector<int>> vseq_offset;
        get_seq_offset(num, seq_offset);
        vseq_offset.push_back(seq_offset);
        in_tensor.set_seq_offset(vseq_offset);
        input.push_back(&in_tensor);
        SequenceConvParam<X86> param(&filter_tensor, context_length, pad_up);
        testbase.set_param(param);
        testbase.add_custom_input(input);
        testbase.run_test(sequence_conv_cpu<float, X86, X86>, 1e-4);
    }
    }
    }
    }
    }

    } while (0);

    LOG(INFO) << "x86 end.......";
#endif
}
int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
