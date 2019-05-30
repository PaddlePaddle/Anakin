#include "saber/core/context.h"
#include "saber/funcs/mvn.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>
#include <cmath>

using namespace anakin::saber;

/**
 * @brief  for each graph, do normalization:
 *          formula:
 *              (x - mean) / ( sqrt(var) + eps )  (the eps iterm avoid divde 0).
 * 
 * @tparam dtype 
 * @tparam TargetType_D 
 * @tparam TargetType_H 
 * @param input 
 * @param output 
 * @param param 
 */
template <typename dtype,typename TargetType_D,typename TargetType_H>
void mvn_cpu_base(const std::vector<Tensor<TargetType_H>* > &input, std::vector<Tensor<TargetType_H>* > &output, MvnParam<TargetType_D> &param) {

    int N = input[0]->num();
    int C = input[0]->channel();
    int H = input[0]->height();
    int W = input[0]->width();

    const dtype* src = (const dtype*)input[0]->data();
    dtype* dst = (dtype*)output[0]->mutable_data();
    int num = N * C;
    int inner_dim = H * W;
    if (param.across_channels) {
        num = N;
        inner_dim *= C; //CHW
    }

    for (int i = 0; i < num; i++) {
        dtype mean = 0;
        dtype std = 0;
        dtype* dst_ptr = dst + i * inner_dim;
        const dtype* src_ptr = src + i * inner_dim;
        //compute mean
        for (int j = 0; j < inner_dim; j++) {
            mean += src_ptr[j];
        }
        mean /= inner_dim;
        //compute variance
        for (int j = 0; j < inner_dim; ++j) {
            std += (src_ptr[j] - mean) * (src_ptr[j] - mean);
        }
        std /= inner_dim;
        std = 1.0f / (sqrtf(std) + param.eps);
        // normalize: (x - mean)/(sqrt(var)+eps)
        if (param.normalize_variance) {
            for (int j = 0; j < inner_dim; j++) {
                dst_ptr[j] = (src_ptr[j] - mean) * std;
            }
        }else { // normalize: x-mean;
            for (int j = 0; j < inner_dim; j++) {
                dst_ptr[j] = src_ptr[j] - mean;
            }
        }
    }
}

template <DataType datatype, typename TargetType_D, typename TargetType_H>
void test_model() {
    using dtype = typename DataTrait<TargetType_H, datatype>::Dtype;

    bool normalize_variance{true};
    bool across_channels{false};
    float eps{1e-9};

    TestSaberBase<TargetType_D, TargetType_H, datatype, Mvn, MvnParam> testbase;
    for(int w_in : {8, 8, 16}) {
        for(int h_in : {2, 8, 32}){
            for(int ch_in : {2, 3, 8, 64}){
                for(int num_in:{1, 21, 32}){
                    Shape shape({num_in, ch_in, h_in, w_in});
                    MvnParam<TargetType_D> param(normalize_variance, across_channels, eps);
                    testbase.set_param(param);
                    testbase.set_rand_limit(-5.0, 5.0);
                    testbase.set_input_shape(shape);
                    if (std::is_same<TargetType_D, MLU>::value) {
                        testbase.run_test(mvn_cpu_base<dtype, TargetType_D, TargetType_H>,
                                          0.02, true);
                    } else {
                        testbase.run_test(mvn_cpu_base<dtype, TargetType_D, TargetType_H>);
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_op_mvn) {
#ifdef USE_CUDA
    test_model<AK_FLOAT, NV, NVHX86>();
#endif
    
#ifdef USE_X86_PLACE
    test_model<AK_FLOAT, X86, X86>();
#endif

#ifdef USE_MLU
//    test_model<AK_FLOAT, MLU, MLUHX86>();
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
