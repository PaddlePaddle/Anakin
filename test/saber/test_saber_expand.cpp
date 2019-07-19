#include "saber/core/context.h"
#include "saber/funcs/expand.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <cmath>

using namespace anakin::saber;


/**
 * @brief   formula: input_shape [n,c, h, w]
 *                   expand_times[n_t, c_t, h_t, w_t].
 *                   out_shape [n*n_t, c*c_t, h*h_t, w*w_t]
 *for example: input: [[a, b], 
                       [c, d]] with shape [2, 2]
               expand_times : [4, 2]
               out: [[a, b, a, b], 
                     [c, d, c, d],
                     [a, b, a, b]
                     [c, d, c, d],
                     [a, b, a, b]
                     [c, d, c, d],
                     [a, b, a, b]
                     [c, d, c, d]]
 *
 * @tparam dtype
 * @tparam TargetType_D
 * @tparam TargetType_H
 * @param input
 * @param output
 * @param param
 */
template <typename dtype, typename TargetType_D, typename TargetType_H>
void expand_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                  std::vector<Tensor<TargetType_H>* >& output, ExpandParam<TargetType_D>& param) {

    int N = input[0]->num();
    int C = input[0]->channel();
    int H = input[0]->height();
    int W = input[0]->width();

    const dtype* src = (const dtype*)input[0]->data();
    dtype* dst = (dtype*)output[0]->mutable_data();
    Shape in_shape = input[0]->valid_shape();
    Shape out_shape  = in_shape;
    int dims = param.expand_times.size();
    for (int i =  param.expand_times.size() - 1; i >= 0; i--) {
        out_shape[i] *= param.expand_times[i];
    }
    output[0]->reshape(out_shape);
    auto out_stride = output[0]->get_stride();
    auto in_stride = input[0]->get_stride();
    for (int out_id = 0; out_id < output[0]->valid_size(); out_id++) {
        int in_id = 0;
        for (int i =  param.expand_times.size() - 1; i >= 0; i--) {
            int in_j = (out_id / out_stride[i]) % in_shape[i];
            in_id += in_j * in_stride[i];
        }
        dst[out_id] = src[in_id];
    }
}

TEST(TestSaberFunc, test_op_expand) {


#ifdef USE_CUDA
    TestSaberBase<NV, NVHX86, AK_FLOAT, Expand, ExpandParam> testbase;

    for (int w_in : {3}) {
        for (int h_in : {4}) {
            for (int ch_in : {2}) {
                for (int num_in : {1}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    std::vector<int> expand_times = {1, 2, 1, 2};
                    ExpandParam<NV> param(expand_times);
                    testbase.set_param(param);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.set_input_shape(shape);
                    testbase.run_test(expand_cpu_base<float, NV, NVHX86>, 2.1e-5f);
                }
            }
        }
    }
#endif

#ifdef USE_X86_PLACE
    TestSaberBase<X86, X86, AK_FLOAT, Expand, ExpandParam> testbase_x86;
    for (int w_in : {3}) {
        for (int h_in : {4}) {
            for (int ch_in : {2}) {
                for (int num_in : {1}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    std::vector<int> expand_times = {1, 2, 1, 2};
                    ExpandParam<X86> param(expand_times);
                    testbase_x86.set_param(param);
                    testbase_x86.set_rand_limit(-5.0, 5.0);
                    testbase_x86.set_input_shape(shape);
                    testbase_x86.run_test(expand_cpu_base<float, X86, X86>, 2.1e-5f);
                }
            }
        }
    }
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
