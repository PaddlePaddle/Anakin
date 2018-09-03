#include <vector>

#include "saber/core/context.h"
#include "saber/funcs/reshape.h"
#include "test_saber_func_pooling_x86.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/x86/kernel/jit_generator.h"
#include "saber/funcs/pooling.h"
#include "x86_test_pool_common_utils.h"
#include "x86_test_common.h"

using namespace anakin::saber;

using pool_test_params_float = pool_test_params;

template <typename data_t>
bool pooling_test(pool_test_params_float &p) {

    test_pool_desc_t pd = p.test_pd;

    std::vector<int> src_dims = {pd.mb, pd.c, pd.ih, pd.iw};
    std::vector<int> dst_dims = {pd.mb, pd.c, pd.oh, pd.ow};

    Shape shape_input(pd.mb, pd.c / 16, pd.ih, pd.iw, 16);
    Tensor5f_C16 input(shape_input);
    fill_tensor_host_rand(input);

    Shape shape_output(pd.mb, pd.c / 16, pd.oh, pd.ow, 16);
    Tensor5f_C16 output(shape_output);
    fill_tensor_host_rand(output);

    Context<X86> ctx_host;
    std::vector<Tensor5f_C16*> inputs(1, &input);
    std::vector<Tensor5f_C16*> outputs(1, &output);

    PoolingParam<Tensor4f> pool_param(pd.kh, pd.kw, pd.padt, pd.padl, pd.strh, pd.strw, p.aalgorithm);
    Pooling<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16> pool;
    if (pool.init(inputs, outputs, pool_param, SPECIFY, SABER_IMPL, ctx_host) != SaberSuccess) {
        LOG(ERROR) << "init failed";
        return false;
    }

    pool(inputs, outputs, pool_param, ctx_host);

    return check_pool_fwd<data_t>(p, inputs, outputs);
}

#define EXPAND_SIZES_2D(mb,ic,ih,iw,oh,ow,kh,kw,padt,padl,strh,strw) \
    4, {mb, ic, 1, ih, iw, 1, oh, ow, 1, kh, kw, 0, padt, padl, 1, strh, strw}

TEST(TestSaberFuncPoolingX86, test_func_pool) {
    Env<X86>::env_init();

    pool_test_params_float test_param [] = {
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D(2, 512, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1)},
        pool_test_params_float{Pooling_average_include_padding,  EXPAND_SIZES_2D( 2, 2048, 7, 7, 1, 1, 7, 7, 0, 0, 1, 1)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 64, 112, 112, 56, 56, 3, 3, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 64, 224, 224, 112, 112, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 128, 112, 112, 56, 56, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 256, 56, 56, 28, 28, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 2, 512, 28, 28, 14, 14, 2, 2, 0, 0, 2, 2)},
        pool_test_params_float{Pooling_max,  EXPAND_SIZES_2D( 6, 512, 14, 14, 7, 7, 2, 2, 0, 0, 2, 2)}
    };

    for (size_t i = 0; i < ARRAY_SIZE(test_param); i++) {
        bool ret = pooling_test<float>(test_param[i]);
        if (ret) {
            LOG(INFO) << "Test Passed";
        }
        else {
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
