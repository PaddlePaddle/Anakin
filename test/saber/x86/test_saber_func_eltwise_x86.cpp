#include <time.h>
#include <stdio.h>

#include "saber/core/context.h"
#include "saber/funcs/eltwise.h"
#include "test_saber_func_eltwise_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "x86_test_common.h"

using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;

struct eltwise_test_params {
    int n;
    int c;
    int h;
    int w; 
    std::vector<float> scale;
    EltwiseType elt_type;
};


template <typename data_t>
bool eltwise_test(eltwise_test_params &p) {
    bool ret = false;
    EltwiseParam<Tensor4f> pm_eltwise;
    pm_eltwise.coeff = p.scale;
    pm_eltwise.operation = p.elt_type;
    // LOG(INFO) << " input size, num=" << num_in << ", channel=" << ch_in << ", height=" << h_in << ", width=" << w_in;

    Shape shape_in(p.n, p.c, p.h, p.w);
    Shape shape_out(p.n, p.c, p.h, p.w);

    std::vector<Tensor4f> src_in(p.scale.size());
    Tensor4f dst_saber, dst_ref;
    
    // src
    for(size_t i = 0; i < src_in.size(); i++) {
        src_in[i].re_alloc(shape_in);
        fill_tensor_host_rand(src_in[i]);
    }

    // referece dst
    float ref_sum = 0;
    dst_ref.re_alloc(shape_out);
    if (p.elt_type == Eltwise_sum) {
        for (int i = 0; i < dst_ref.size(); i++) {
            ref_sum = 0;
            for (size_t j = 0; j < src_in.size(); j++) {
                ref_sum += src_in[j].mutable_data()[i] * p.scale[j];
            }
            dst_ref.mutable_data()[i] = ref_sum;
        }
    }

    // saber dst
    Context<X86> ctx_host;
    std::vector<Tensor4f*> input_eltwise;
    std::vector<Tensor4f*> output_eltwise;

    for (size_t i = 0; i < src_in.size(); i++) {
        input_eltwise.push_back(&src_in[i]);
    }

    Eltwise<X86, AK_FLOAT> op_eltwise;
    dst_saber.re_alloc(shape_out);
    output_eltwise.push_back(&dst_saber);
    op_eltwise.compute_output_shape(input_eltwise, output_eltwise, pm_eltwise);

    if (op_eltwise.init(input_eltwise, output_eltwise, pm_eltwise, SPECIFY, SABER_IMPL, ctx_host) == SaberSuccess) {
        op_eltwise(input_eltwise, output_eltwise, pm_eltwise, ctx_host);
        ret = compare_tensor(dst_saber, dst_ref, 1e-6);
    } else {
        ret = true;
    }
    return ret;
}


TEST(TestSaberEltwiseX86, test_tensor_eltwise) {
    Env<X86>::env_init();
    eltwise_test_params test_param [] = {
        eltwise_test_params{1, 256, 56, 56, {1.0f, 1.0f}, Eltwise_sum},
        eltwise_test_params{1, 256, 56, 56, {2.0f, 3.0f}, Eltwise_sum},
        eltwise_test_params{2, 512, 28, 28, {1.0f, 1.0f}, Eltwise_sum},
        eltwise_test_params{4, 1024, 14, 14, {1.0f, 1.0f}, Eltwise_sum},
        eltwise_test_params{4, 1024, 14, 14, {1.0f, 1.0f}, Eltwise_max},
        eltwise_test_params{8, 2048, 7, 7, {1.0f, 1.0f}, Eltwise_sum}
    };

    size_t test_num = sizeof(test_param) / sizeof(test_param[0]);
    for (size_t i = 0; i < test_num; i++) {
        bool ret = eltwise_test<float>(test_param[i]);
        if (ret) {
            LOG(INFO) << "Test Passed";
        }
        else {
            LOG(ERROR) << "Test Failed";
        }
    }
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

