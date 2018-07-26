#include <vector>

#include "saber/core/context.h"
#include "saber/funcs/reshape.h"
#include "test_saber_func_conv_act_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/conv_act.h"

#include "x86_test_common.h"
#include "utils/logger/logger.h"

using namespace anakin::saber;

typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;
typedef Tensor<X86, AK_FLOAT, NCHW_C16> Tesnor5f_C16;
typedef Tensor<X86, AK_FLOAT, NCHW_C16> Tesnor5f_C8;

#include "x86_test_conv_common_utils.h"

enum LayoutType {
   Layout_Invalid = 0,
   Layout_NCHW = 1,
   Layout_NCHW_C16 = 2,
   Layout_NCHW_C8 = 3,
};

struct conv_act_params {
   LayoutType weight_type;
   LayoutType input_type;
   LayoutType output_type;
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

template <typename LayOutType_op, typename LayOutType_in, typename LayOutType_out>
bool saber_conv_act(const conv_act_params &p,
                  Tensor4f &input_ref,
                  Tensor4f &weight,
                  Tensor4f &bias,
                  Tensor4f &output_ref) {
    Context<X86> ctx_host;
    typedef Tensor<X86, AK_FLOAT, LayOutType_in> inTensor;
    typedef Tensor<X86, AK_FLOAT, LayOutType_out> outTensor;
    typedef Tensor<X86, AK_FLOAT, LayOutType_op> opTensor;
    
    Shape shape_input, shape_output, shape_weight, shape_bias;
    if (std::is_same<LayOutType_in, NCHW>::value && std::is_same<LayOutType_out, NCHW_C16>::value) {
        Shape shape_input_tmp(p.n, p.ic, p.ih, p.iw);
        Shape shape_output_tmp(p.n, p.oc / 16, p.oh, p.ow, 16);
        Shape shape_weight_tmp(p.oc, p.ic, p.kh, p.kw);
        Shape shape_bias_tmp(p.oc, 1, 1, 1);
        shape_input = shape_input_tmp;
        shape_output = shape_output_tmp;
        shape_weight = shape_weight_tmp;
        shape_bias = shape_bias_tmp;
      } else if (std::is_same<LayOutType_in, NCHW_C16>::value && std::is_same<LayOutType_out, NCHW_C16>::value) {
        Shape shape_input_tmp(p.n, p.ic / 16, p.ih, p.iw, 16);
        Shape shape_output_tmp(p.n, p.oc / 16, p.oh, p.ow, 16);
        Shape shape_weight_tmp(p.oc, p.ic, p.kh, p.kw);
        Shape shape_bias_tmp(p.oc, 1, 1, 1);
        shape_input = shape_input_tmp;
        shape_output = shape_output_tmp;
        shape_weight = shape_weight_tmp;
        shape_bias = shape_bias_tmp;
      } else if (std::is_same<LayOutType_in, NCHW>::value && std::is_same<LayOutType_out, NCHW_C8>::value) {
        Shape shape_input_tmp(p.n, p.ic, p.ih, p.iw);
        Shape shape_output_tmp(p.n, p.oc / 8, p.oh, p.ow, 8);
        Shape shape_weight_tmp(p.oc, p.ic, p.kh, p.kw);
        Shape shape_bias_tmp(p.oc, 1, 1, 1);
        shape_input = shape_input_tmp;
        shape_output = shape_output_tmp;
        shape_weight = shape_weight_tmp;
        shape_bias = shape_bias_tmp;
    }

    Shape shape_output_nchw(p.n, p.oc, p.oh, p.ow);

    inTensor input(shape_input);
    if (std::is_same<LayOutType_in, NCHW_C16>::value) {
       reorder<Tensor4f, inTensor>(input_ref, input);
    } else {
        for (int i = 0; i < input_ref.size(); ++i) {
            *(input.mutable_data() + i) = *(input_ref.data() + i);
        }
    }
    std::vector<inTensor*> inputs(1, &input);

    outTensor output(shape_output);
    fill_tensor_host_const(output, 0);
    std::vector<outTensor*> outputs(1, &output);

    Tensor4f output_nchw(shape_output_nchw);
    fill_tensor_host_const(output_nchw, 0);

    ConvParam<opTensor> conv_param(p.g, 
            p.pad_h, p.pad_w, 
            p.stride_h, p.stride_w, 
            p.dil_w, p.dil_h,
            &weight, &bias, p.alpha, p.beta);

    ActivationParam<opTensor> act_param(Active_relu);

    ConvActiveParam<opTensor> conv_act_param(conv_param, act_param);

    ConvAct<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, LayOutType_op, LayOutType_in, LayOutType_out> conv_act_op;
    SaberStatus status = conv_act_op.init(inputs, outputs, conv_act_param, SPECIFY, SABER_IMPL, ctx_host);
    if (status == SaberSuccess) {
       conv_act_op(inputs, outputs, conv_act_param, ctx_host);
       reorder<outTensor, Tensor4f>(*(outputs[0]), output_nchw);
       return compare_tensor<Tensor4f>(output_nchw, output_ref);
    } else {
       LOG(ERROR) << "/*----------Init Failed!----------*/";
       return false;
    }
}


template <typename dtype>
bool conv_act_test(conv_act_params &p) {
    // create reference Tensor and Param
    Shape shape_input(p.n, p.ic, p.ih, p.iw);
    Tensor4f input_ref(shape_input);
    fill_tensor_host_rand(input_ref);
    std::vector<Tensor4f *> inputs_ref(1, &input_ref);

    Shape shape_output(p.n, p.oc, p.oh, p.ow);
    Tensor4f output_ref(shape_output);  
    fill_tensor_host_const(output_ref, 0);
    std::vector<Tensor4f *> outputs_ref(1, &output_ref);

    Shape shape_weight(p.oc, p.ic, p.kh, p.kw);
    Tensor4f weight(shape_weight);
    fill_tensor_host_rand(weight);

    Shape shape_bias(p.oc, 1, 1, 1);
    Tensor4f bias(shape_bias);
    fill_tensor_host_rand(bias);

    ConvParam<Tensor4f> conv_param(p.g, 
            p.pad_h, p.pad_w, 
            p.stride_h, p.stride_w, 
            p.dil_w, p.dil_h,
            &weight, &bias, p.alpha, p.beta);

    ActivationParam<Tensor4f> act_param(Active_relu);

    // compute reference
    compute_ref_conv_relu_fwd<AK_FLOAT, NCHW>(inputs_ref, outputs_ref, &conv_param, &act_param);

    // compute saber 
    if (p.input_type == Layout_NCHW && p.output_type == Layout_NCHW_C16) {
        return saber_conv_act<NCHW, NCHW, NCHW_C16>(p, input_ref, weight, bias, output_ref);
    } else if (p.input_type == Layout_NCHW_C16 && p.output_type == Layout_NCHW_C16) {
        return saber_conv_act<NCHW, NCHW_C16, NCHW_C16>(p, input_ref, weight, bias, output_ref);
    } else if (p.input_type == Layout_NCHW && p.output_type == Layout_NCHW_C8) { 
        return saber_conv_act<NCHW, NCHW, NCHW_C8>(p, input_ref, weight, bias, output_ref);
    }
    return false;
}

using conv_act_params_float = conv_act_params;


TEST(TestSaberFuncConvActX86, test_func_conv_act) {
    Env<X86>::env_init();
   
    conv_act_params_float test_param[] = {
        conv_act_params_float{Layout_NCHW, Layout_NCHW, Layout_NCHW_C8,
            1, 1, 3, 224, 224, 64, 224, 224, 3, 3, 1, 1, 1, 1, 0, 0,          // first layer of VGG-16, AVX2
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW, Layout_NCHW_C8,
            1, 1, 3, 224, 224, 64, 112, 112, 7, 7, 3, 3, 2, 2, 0, 0,            // first layer of ResNet-50, AVX2
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW, Layout_NCHW_C16,
            1, 1, 3, 224, 224, 64, 224, 224, 3, 3, 1, 1, 1, 1, 0, 0,          // first layer of VGG-16, AVX512
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW, Layout_NCHW_C16,
            1, 1, 3, 224, 224, 64, 112, 112, 7, 7, 3, 3, 2, 2, 0, 0,            // first layer of ResNet-50, AVX512
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW_C16, Layout_NCHW_C16,  // non-first layer of VGG16
            1, 1, 32, 3, 3, 32, 3, 3, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW_C16, Layout_NCHW_C16,
            1, 1, 64, 112, 112, 128, 112, 112, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW_C16, Layout_NCHW_C16,
            1, 1, 128, 56, 56, 256, 56, 56, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW_C16, Layout_NCHW_C16,
            1, 1, 256, 56, 56, 256, 56, 56, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW_C16, Layout_NCHW_C16,
            1, 1, 256, 28, 28, 512, 28, 28, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW_C16, Layout_NCHW_C16,
            1, 1, 512, 28, 28, 512, 28, 28, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW_C16, Layout_NCHW_C16,
            1, 1, 512, 14, 14, 512, 14, 14, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1) },
        conv_act_params_float{Layout_NCHW, Layout_NCHW_C16, Layout_NCHW_C16,
            1, 1, 512, 14, 14, 512, 14, 14, 3, 3, 1, 1, 1, 1, 0, 0,
            float(1), float(0), float(0), float(1) }
    };

    for (size_t i = 0; i < ARRAY_SIZE(test_param); ++i) {
        bool flag = conv_act_test<float>(test_param[i]);
        if (flag) {
            LOG(INFO) << "Test Passed";
        } else {
            LOG(ERROR) << "Test Failed";
        }
    } 
}

int main(int argc, const char** argv) {
#if 0
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
#endif
    return 0;
}
