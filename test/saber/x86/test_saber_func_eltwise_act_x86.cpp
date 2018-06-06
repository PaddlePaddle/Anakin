#include <time.h>
#include <stdio.h>

#include "saber/core/context.h"
#include "saber/funcs/eltwise_act.h"
#include "test_saber_func_eltwise_act_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "x86_test_common.h"

using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;

struct eltwise_act_test_params {
    int n;
    int c;
    int h;
    int w; 
    std::vector<float> scale;
    bool act;
    ActiveType act_type;
    EltwiseType elt_type;
};

template <typename data_t>
bool eltwise_act_test(eltwise_act_test_params &p) {
    bool ret = false;

    EltwiseActiveParam<Tensor4f> pm_eltwise_act;
    ActivationParam<Tensor4f> pm_act(p.act_type);
    pm_eltwise_act.eltwise_param.coeff = p.scale;
    pm_eltwise_act.eltwise_param.operation = p.elt_type;
    pm_eltwise_act.has_activation = p.act;
    pm_eltwise_act.activation_param = pm_act;
    // LOG(INFO) << " input size, num=" << num_in << ", channel=" << ch_in << ", height=" << h_in << ", width=" << w_in;

    Shape shape_in(p.n, p.c, p.h, p.w);
    Shape shape_out(p.n, p.c, p.h, p.w);

    std::vector<Tensor4f> src_in(p.scale.size());
    Tensor4f dst_saber, dst_ref;
    
    // fill src
    for (size_t i = 0; i < src_in.size(); i++) {
        src_in[i].re_alloc(shape_in);
        fill_tensor_host_rand(src_in[i]);
    }

    // reference dst
    float ref_sum = 0.f;
    dst_ref.re_alloc(shape_out);
    if (p.elt_type == Eltwise_sum) {
        for (int i = 0; i < dst_ref.size(); i++) {
            ref_sum = 0.f;
            for (size_t j = 0; j < src_in.size(); j++) {
                ref_sum += src_in[j].mutable_data()[i] * p.scale[j];
            }
            dst_ref.mutable_data()[i] = ref_sum;
        }

        // relu
        if (p.act) {
            if (p.act_type == Active_relu) {
                for (int i = 0; i < dst_ref.size(); i++){
                    if (dst_ref.mutable_data()[i] < 0) {
                        dst_ref.mutable_data()[i] = 0;
                    }
                }
            }
        }
    }

    // saber dst
    Context<X86> ctx_host;

    std::vector<Tensor4f*> input_eltwise_act;
    std::vector<Tensor4f*> output_eltwise_act;

    for(size_t i = 0; i < src_in.size(); i++) {
        input_eltwise_act.push_back(&src_in[i]);
    }

    EltwiseActive<X86, AK_FLOAT> op_eltwise_act;

    dst_saber.re_alloc(shape_out);
    output_eltwise_act.push_back(&dst_saber);
    op_eltwise_act.compute_output_shape(input_eltwise_act, output_eltwise_act, pm_eltwise_act);

    if (op_eltwise_act.init(input_eltwise_act, output_eltwise_act, pm_eltwise_act, SPECIFY, SABER_IMPL, ctx_host) == SaberSuccess) {
        op_eltwise_act(input_eltwise_act, output_eltwise_act, pm_eltwise_act, ctx_host);
        ret = compare_tensor<Tensor4f>(dst_ref, dst_saber, 1e-6); 
    } else {
        //LOG(INFO) << "op_eltwise_act init fail due to wrong parameter type";
        ret = true;
    }
    return ret;
}


TEST(TestSaberEltwiseActX86, test_tensor_eltwise_act) {
    Env<X86>::env_init();
    eltwise_act_test_params test_param [] = {
        eltwise_act_test_params{1, 256, 56, 56, {1.0f, 1.0f}, true, Active_relu, Eltwise_sum},
        eltwise_act_test_params{1, 256, 56, 56, {2.0f, 3.0f}, true, Active_relu, Eltwise_sum},
        eltwise_act_test_params{2, 512, 28, 28, {1.0f, 1.0f}, true, Active_relu, Eltwise_sum},
        eltwise_act_test_params{2, 512, 28, 28, {1.0f, 1.0f}, true, Active_relu, Eltwise_max},
        eltwise_act_test_params{4, 1024, 14, 14, {1.0f, 1.0f}, true, Active_relu, Eltwise_sum},
        eltwise_act_test_params{4, 1024, 14, 14, {1.0f, 1.0f}, false, Active_relu, Eltwise_sum},
        eltwise_act_test_params{8, 2048, 7, 7, {1.0f, 1.0f}, true, Active_elu, Eltwise_sum},
        eltwise_act_test_params{8, 2048, 7, 7, {1.0f, 1.0f}, true, Active_relu, Eltwise_sum}
    };

    size_t test_num = sizeof(test_param) / sizeof(test_param[0]);
    for (size_t i = 0; i < test_num; i++) {
        bool ret = eltwise_act_test<float>(test_param[i]);
        if (ret) {
            LOG(INFO) << "Tests Passed";
        }
        else {
            LOG(ERROR) << "Tests Failed";
        }
    }
}

int main(int argc, const char** argv) {
    // initialize logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
