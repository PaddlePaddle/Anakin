#include <vector>

#include "saber/core/context.h"
#include "saber/funcs/reshape.h"
#include "test_saber_func_conv_act_pool_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/conv_act_pooling.h"
#include "x86_test_common.h"

using namespace anakin::saber;
#include "x86_test_pool_common_utils.h"
#include "x86_test_conv_common_utils.h"

struct conv_act_params {
   int n, g;
   int ic, ih, iw;
   int oc, oh, ow;
   int kh, kw;
   int pad_h, pad_w;
   int stride_h, stride_w;
   int dil_h, dil_w;
   float alpha, beta;
   float negative_slope;
   float coef;
};

struct conv_act_pool_params {
   conv_act_params conv_p;
   pool_test_params pool_p;
};

template <typename dtype>
bool conv_act_pool_test(conv_act_pool_params &c_p_param) {
    conv_act_params conv_p = c_p_param.conv_p;
    pool_test_params pool_p = c_p_param.pool_p;
    test_pool_desc_t pd = pool_p.test_pd;

    // init parameters
    Shape shape_weight(pd.c, conv_p.ic, conv_p.kh, conv_p.kw);
    Tensor4f weight(shape_weight);
    fill_tensor_host_rand(weight);

    Shape shape_bias(pd.c, 1, 1, 1);
    Tensor4f bias(shape_bias);
    fill_tensor_host_rand(bias);

    ConvParam<Tensor4f> conv_param(conv_p.g,
            conv_p.pad_h, conv_p.pad_w, 
            conv_p.stride_h, conv_p.stride_w, 
            conv_p.dil_w, conv_p.dil_h,
            &weight, &bias, conv_p.alpha, conv_p.beta);

    ActivationParam<Tensor4f> act_param_ref(Active_relu);

    // init reference data
    Shape shape_input_ref(conv_p.n, conv_p.ic, conv_p.ih, conv_p.iw);
    Tensor4f input_ref(shape_input_ref);
    fill_tensor_host_const(input_ref, 1);
    std::vector<Tensor4f*> inputs_ref(1, &input_ref);

    Shape conv_shape_output_ref(conv_p.n, conv_p.oc, conv_p.oh, conv_p.ow);
    Shape conv_shape_output_ref_c16(conv_p.n, conv_p.oc / 16, conv_p.oh, conv_p.ow, 16);
    Tensor4f output_ref(conv_shape_output_ref);
    Tensor5f_C16 output_ref_c16(conv_shape_output_ref_c16);
    fill_tensor_host_const(output_ref, 0);
    fill_tensor_host_const(output_ref_c16, 0);
    std::vector<Tensor4f*> outputs_ref(1, &output_ref);
    std::vector<Tensor5f_C16*> outputs_ref_c16(1, &output_ref_c16);

    Context<X86> ctx_host;
    Shape shape_input(conv_p.n, conv_p.ic / 16, conv_p.ih, conv_p.iw, 16);
    Tensor5f_C16 input(shape_input);
    std::vector<Tensor5f_C16*> inputs(1, &input);

    reorder<Tensor4f, Tensor5f_C16>(*inputs_ref[0], *inputs[0]);

    Shape shape_output(pd.mb, pd.c / 16, pd.oh, pd.ow, 16);
    Tensor5f_C16 output(shape_output);
    fill_tensor_host_const(output, 0);
    std::vector<Tensor5f_C16*> outputs(1, &output);

    ActivationParam<Tensor4f> act_param(Active_relu);
    PoolingParam<Tensor4f> pool_param(pd.kh, pd.kw, pd.padt, pd.padl, pd.strh, pd.strw, pool_p.aalgorithm);
    ConvActivePoolingParam<Tensor4f> conv_act_pool_param(conv_param, act_param, pool_param);

    // reference
    compute_ref_conv_relu_fwd<AK_FLOAT, NCHW>(inputs_ref, outputs_ref, &conv_param, &act_param);
    reorder<Tensor4f, Tensor5f_C16>(*outputs_ref[0], *outputs_ref_c16[0]);

    // saber 
    ConvActPooling<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16> conv_act_pooling_op;
    if (conv_act_pooling_op.init(inputs, outputs, conv_act_pool_param, SPECIFY, SABER_IMPL, ctx_host) != SaberSuccess) {
        LOG(ERROR) << "init failed";
        return false;
    }
    conv_act_pooling_op(inputs, outputs, conv_act_pool_param, ctx_host);

    return check_pool_fwd<float>(pool_p, outputs_ref_c16, outputs);
}

using conv_act_pool_params_float = conv_act_pool_params;

TEST(TestSaberFuncConvActPoolX86, test_func_conv_act_pool) {
    Env<X86>::env_init();

    conv_act_pool_params_float test_param[] = {
        conv_act_pool_params_float{{1, 1, 16, 3, 3, 16, 3, 3, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1)},{Pooling_max, 4,{1, 16, 1, 3, 3, 1, 1, 1, 1, 3, 3, 0, 0, 0, 1, 1, 1}, true, SaberSuccess}},
        conv_act_pool_params_float{{1, 1, 512, 28, 28, 512, 28, 28, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1)},{Pooling_max, 4,{1, 512, 1, 28, 28, 1, 14, 14, 1, 2, 2, 0, 0, 0, 1, 2, 2}, true, SaberSuccess}},
        conv_act_pool_params_float{{1, 1, 128, 112, 112, 128, 112, 112, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1)},{Pooling_max, 4,{1, 128, 1, 112, 112, 1, 56, 56, 1, 2, 2, 0, 0, 0, 1, 2, 2}, true, SaberSuccess}},
        conv_act_pool_params_float{{1, 1, 512, 14, 14, 512, 14, 14, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1)},{Pooling_max, 4,{1, 512, 1, 14, 14, 1, 7, 7, 1, 2, 2, 0, 0, 0, 1, 2, 2}, true, SaberSuccess}},
        conv_act_pool_params_float{{1, 1, 64, 224, 224, 64, 224, 224, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1)},{Pooling_max, 4,{1, 64, 1, 224, 224, 1, 112, 112, 1, 2, 2, 0, 0, 0, 1, 2, 2}, true, SaberSuccess}},
        conv_act_pool_params_float{{1, 1, 256, 56, 56, 256, 56, 56, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1)},{Pooling_max, 4,{1, 256, 1, 56, 56, 1, 28, 28, 1, 2, 2, 0, 0, 0, 1, 2, 2}, true, SaberSuccess}}, 
    };

    for (size_t i = 0; i < ARRAY_SIZE(test_param); ++i) {
        LOG(INFO) << "case " << i << ":";
        bool flag = conv_act_pool_test<float>(test_param[i]);
        if (flag) {
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

