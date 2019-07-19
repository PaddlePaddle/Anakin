#include "saber/core/context.h"
#include "saber/funcs/mat_mul.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>

using namespace anakin::saber;

/**
 * @brief matrix transpose.
 *
 * @tparam dtype : data type.
 * @tparam TargetType : device type.
 * @param data: input/output
 */
template<typename dtype, typename TargetType>
void batch_transpos(const Tensor<TargetType>& data, Tensor<TargetType>& trans_data) {
    int M = data.height();
    int N = data.width();
    int B = data.num() * data.channel();
    const dtype* data_ptr = (const dtype*)data.data();
    dtype* trans_data_ptr = (dtype*)trans_data.mutable_data();

    for (int b = 0; b < B; b++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                trans_data_ptr[b * M * N + n * M + m] = data_ptr[b * M * N + m * N + n];
            }
        }
    }

    trans_data.set_height(N);
    trans_data.set_width(M);
}


/**
 * @brief mat_mul compute (native cpu version)
 *
 * @tparam dtype
 * @tparam TargetType_D
 * @tparam TargetType_H
 * @param input: input[0] with size M*K, input[1] with N*K
 * @param output: output = input[0] * input[1].
 * @param param
 */
template <typename dtype, typename TargetType_D, typename TargetType_H>
void mat_mul_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                      std::vector<Tensor<TargetType_H>* >& output, MatMulParam<TargetType_D>& param) {

    int M, N, K, B;
    //bath size
    B = input[0]->num() * input[0]->channel();

    Tensor<TargetType_H> trans_input0(input[0]->valid_shape());
    trans_input0.copy_from(*input[0]);
    Tensor<TargetType_H> trans_input1(input[1]->valid_shape());
    trans_input1.copy_from(*input[1]);

    //whether input[0] trans.
    if (param._is_transpose_X) {
        batch_transpos<dtype, TargetType_H>(*input[0], trans_input0);
    }

    //whether input[1] trans
    if (param._is_transpose_Y) {
        batch_transpos<dtype, TargetType_H>(*input[1], trans_input1);
    }

    CHECK_EQ(trans_input0.width(), trans_input1.height()) << "can't do matrix multiplay";

    M = trans_input0.height();
    N = trans_input1.width();
    K = trans_input1.height();

    dtype* out_ptr = (dtype*)output[0]->mutable_data();
    const dtype* in_ptr0 = (const dtype*)trans_input0.data();
    const dtype* in_ptr1 = (const dtype*)trans_input1.data();

    for (int b = 0; b < B; b++) {
        float* optr = out_ptr + b * M * N;
        const float* iptr0 = in_ptr0 + b * M * K;
        const float* iptr1 = in_ptr1 + b * K * N;

        for (int i = 0; i < M; ++i) {
            float* pdout = optr + i * N;

            for (int j = 0; j < N; ++j) {
                pdout[j] = 0;

                for (int l = 0; l < K; ++l) {
                    pdout[j] += iptr0[i * K + l] * iptr1[l * N + j];
                }
            }
        }
    }

}

template <DataType datatype, typename TargetType_D, typename TargetType_H>
void test_model() {
    using dtype = typename DataTrait<TargetType_H, datatype>::Dtype;

    bool trans_a = true;
    bool trans_b = false;
    int input_num = 2;
    TestSaberBase<TargetType_D, TargetType_H, datatype, MatMul, MatMulParam> testbase(input_num);

    std::vector<int> v_w_in   = {2, 8, 16};
    std::vector<int> v_h_in   = {2, 8, 32};
    std::vector<int> v_ch_in  = {2, 3, 8, 64};
    std::vector<int> v_num_in = {1, 21, 32};

    // mlu mat_mul is too slow when b is large
    if (std::is_same<TargetType_D, MLU>::value) {
        v_ch_in  = {3, 8};
        v_num_in = {2, 4};
    }

    for (int w_in : v_w_in) {
        for (int h_in : v_h_in) {
            for (int ch_in : v_ch_in) {
                for (int num_in : v_num_in) {
                    Shape shape0 = trans_a ? Shape({num_in, ch_in, w_in, h_in}) :
                                             Shape({num_in, ch_in, h_in, w_in});
                    Shape shape1 = trans_b ? Shape({num_in, ch_in, h_in, w_in}) :
                                             Shape({num_in, ch_in, w_in, h_in});
                    std::vector<Shape> shapes;
                    shapes.push_back(shape0);
                    shapes.push_back(shape1);
                    MatMulParam<TargetType_D> param(trans_a, trans_b);
                    testbase.set_param(param);
                    testbase.set_rand_limit(1, 12);
                    testbase.set_input_shape(shapes);
                    if (std::is_same<TargetType_D, MLU>::value) {
                        testbase.run_test(mat_mul_cpu_base<dtype, TargetType_D, TargetType_H>, 
                                          0.005, true);
                    } else {
                        testbase.run_test(mat_mul_cpu_base<dtype, TargetType_D, TargetType_H>);
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_op_mat_mul) {
#ifdef USE_CUDA
    test_model<AK_FLOAT, NV, NVHX86>();
#endif

#ifdef USE_X86_PLACE
    test_model<AK_FLOAT, X86, X86>();
#endif

#ifdef USE_MLU
    test_model<AK_FLOAT, MLU, MLUHX86>();
#endif
}


int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
