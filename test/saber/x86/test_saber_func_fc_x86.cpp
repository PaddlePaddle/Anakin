#include <vector>
#include <iterator>
#include "saber/core/context.h"
#include "saber/funcs/fc.h"
#include "test_saber_func_fc_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "x86_test_common.h"

using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;
typedef Tensor<X86, AK_FLOAT, HW> Tensor2f;
typedef Tensor<X86, AK_FLOAT, W> Tensor1f;
#define EXPAND_SIZES_2D(mb, ic, oc, kh, kw) {mb, ic, oc, kh, kw}

void compute_ref_inner_product_fwd(Tensor4f &src, Tensor4f &dst, FcParam<Tensor4f> &param) {
    typedef typename Tensor4f::Dtype data_t;

    const data_t *src_data = static_cast<const data_t*>(src.get_buf()->get_data());
    const data_t *weights_data = static_cast<const data_t*>(param.weights->get_buf()->get_data());
    const data_t *bias_data = NULL;
    if (param.bias) {
        bias_data = static_cast<const data_t*>(param.bias->get_buf()->get_data());
    }
    data_t *dst_data = static_cast<data_t*>(dst.get_buf()->get_data_mutable());

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < src.num(); n++) {
        for (int oc = 0; oc < dst.channel(); oc++) {
            int oidx = n * dst.channel() + oc;
            dst_data[oidx] = bias_data ? bias_data[oc] : data_t{0};
            for (int ic = 0; ic < src.channel(); ic++) {
                for (int kh = 0; kh < src.height(); kh++)
                for (int kw = 0; kw < src.width(); kw++) {
                    int iidx = n * src.channel() * src.height() * src.width() + ic * src.height() * src.width() + kh * src.width() + kw;
                    int widx = oc * src.channel() * src.height() * src.width() + ic * src.height() * src.width() + kh * src.width() + kw;
                    dst_data[oidx] += src_data[iidx] * weights_data[widx];
                }
            }
        }
    }
}

struct test_inner_product_descr_t {
    int mb;
    int ic;
    int oc;
    int kh, kw;
};

struct inprod_test_params {
    test_inner_product_descr_t test_ipd;
    bool with_bias;
    bool expect_to_fail;
};

using inprod_test_params_float = inprod_test_params;

template <typename data_t>
bool inner_product_test(inprod_test_params& p) {
    test_inner_product_descr_t ipd = p.test_ipd;

    bool with_bias = p.with_bias;

    Shape inputShape(ipd.mb, ipd.ic, ipd.kh, ipd.kw);
    Tensor4f saberInput(inputShape);
    fill_tensor_host_rand<Tensor4f>(saberInput);

    Shape weightShape(ipd.oc, ipd.ic, ipd.kh, ipd.kw);
    Tensor4f saberWeight(weightShape);
    fill_tensor_host_rand(saberWeight);
    
    std::vector<Tensor4f*> input_host_4d;
    input_host_4d.push_back(&saberInput);

    Tensor4f saberOutput;
    std::vector<Tensor4f*> output_host_4d;
    output_host_4d.push_back(&saberOutput);

    FcParam<Tensor4f> param_host_4d(&saberWeight, ipd.oc);

    Shape biasShape(1, 1, 1, ipd.oc);
    Tensor4f saberBias(biasShape);
    if (with_bias) {
        fill_tensor_host_rand<Tensor4f>(saberBias);
        param_host_4d.bias = &saberBias;
    }

    // get reference result
    Tensor4f refOutput;
    compute_ref_inner_product_fwd(saberInput, refOutput, param_host_4d);

    Fc<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> saberFc;
    saberFc.compute_output_shape(input_host_4d, output_host_4d, param_host_4d);
    saberOutput.re_alloc(saberOutput.shape());
    refOutput.re_alloc(saberOutput.shape());

    // get reference result
    compute_ref_inner_product_fwd(saberInput, refOutput, param_host_4d);

    // get saber result
    Context<X86> ctx_host;
    saberFc.init(input_host_4d, output_host_4d, param_host_4d, SPECIFY, VENDER_IMPL, ctx_host);
    saberFc(input_host_4d, output_host_4d, param_host_4d, ctx_host);

    bool ret = false;
    ret = compare_tensor<Tensor4f>(saberOutput, refOutput);
    return ret;
}

TEST(TestSaberFuncFcX86, test_gemm_fc_bias) {
    Env<X86>::env_init();
    inprod_test_params_float test_param[] = {
        inprod_test_params_float{ EXPAND_SIZES_2D(2, 32, 48, 6, 6), true },
        inprod_test_params_float{ EXPAND_SIZES_2D(2, 512, 48, 2, 2), true},
        inprod_test_params_float{ EXPAND_SIZES_2D(2, 32, 48, 6, 6), true},
        inprod_test_params_float{ EXPAND_SIZES_2D(2, 32, 1152, 1, 1), true },
        inprod_test_params_float{ EXPAND_SIZES_2D(2, 2, 4, 1, 1), true }
    };

    for (size_t i = 0; i < ARRAY_SIZE(test_param); i++) {
        bool flag = false;
        flag = inner_product_test<float>(test_param[i]);
        if (flag) {
            LOG(INFO) << "Test Passed";
        } else {
            LOG(ERROR) << "Test Failed";
        }
    }
}

TEST(TestSaberFuncFcX86, test_gemm_fc_nobias) {
    Env<X86>::env_init();

    inprod_test_params_float test_param[] = {
        inprod_test_params_float{ EXPAND_SIZES_2D(2, 32, 48, 6, 6), false },
        inprod_test_params_float{ EXPAND_SIZES_2D(2, 512, 48, 2, 2), false },
        inprod_test_params_float{ EXPAND_SIZES_2D(2, 32, 48, 6, 6), false },
        inprod_test_params_float{ EXPAND_SIZES_2D(2, 32, 1152, 1, 1), false },
        inprod_test_params_float{ EXPAND_SIZES_2D(2, 2, 4, 1, 1), false }
    };

    for (size_t i = 0; i < ARRAY_SIZE(test_param); i++) {
        bool flag = false;
        flag = inner_product_test<float>(test_param[i]);
        if (flag) {
            LOG(INFO) << "Test Passed";
        } else {
            LOG(ERROR) << "Test Failed";
        }
    }
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
