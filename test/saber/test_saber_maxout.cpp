#include "saber/core/context.h"
#include "saber/funcs/maxout.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <cfloat>

using namespace anakin::saber;

template <typename dtype, typename TargetType_D, typename TargetType_H>
void maxout_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                      std::vector<Tensor<TargetType_H>* >& output, MaxOutParam<TargetType_D>& param) {
    
    int group = param.groups;
    int batch_size = input[0]->num();
    int channel = input[0]->channel() / group;
    int height = output[0]->height();
    int width = input[0]->width();

    int feature_size = height * width;
    int feature_map_size = feature_size * channel;

    const dtype* input_ptr = (const dtype*)input[0]->data();
    dtype* output_ptr = (dtype*)output[0]->mutable_data();

    for (int i = 0; i < batch_size; i++) {
        int n_id = i * feature_map_size;
        for (int c = 0; c < channel; c++) {
            int c_id = c * feature_size;
            for (int f = 0; f < feature_size; f++) {
                dtype max = static_cast<dtype>(-FLT_MAX);
                for (int g = 0; g < group; g++) {
                    dtype tmp = input_ptr[(n_id + c_id) * group + g * feature_size + f];
                    max = max < tmp ? tmp : max; 
                }
                output_ptr[n_id + c_id + f] = max;
            }
        }
    }
}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_maxout() {
    int groups = 2;
    TestSaberBase<TargetType_D, TargetType_H, Dtype, MaxOut, MaxOutParam> testbase;
    MaxOutParam<TargetType_D> param(groups);

    for (int w_in : {8, 8, 16}) {
        for (int h_in : {2, 8, 32}) {
            for (int ch_in : {3, 4, 8, 64}) {
                for (int num_in:{1, 21, 32}) {
                    Shape shape({num_in, ch_in, h_in, w_in});
                    testbase.set_param(param);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.set_input_shape(shape);
                    testbase.run_test(maxout_cpu_base<float, TargetType_D, TargetType_H>);
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_op_maxout) {

#ifdef USE_CUDA
   //Init the test_base
    test_maxout<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    test_maxout<AK_FLOAT, X86, X86>();
#endif
#ifdef USE_ARM_PLACE
    //test_maxout<AK_FLOAT, ARM, ARM>();
#endif
#ifdef USE_BM
   // Env<BM>::env_init();
    //test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
