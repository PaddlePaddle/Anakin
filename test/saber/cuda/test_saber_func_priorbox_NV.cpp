
#include "core/context.h"
#include "funcs/priorbox.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncNV, test_func_priorbox_NV) {

    typedef TargetWrapper<NV> API;

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef TensorDf4::Dtype dtype;

    int test_iter = 100;

    int width = 300;
    int height = 300;
    int channel = 3;
    int num = 1;
    int w_fea = 19;
    int h_fea = 19;
    int c_fea = 512;

    LOG(INFO) << " input data size, num=" << num << ", channel=" << \
              channel << ", height=" << height << ", width=" << width;

    LOG(INFO) << " input feature tensor size, num=" << num << ", channel=" << \
              c_fea << ", height=" << h_fea << ", width=" << w_fea;
    //! create input output tensor
    Shape sh_fea{num, c_fea, h_fea, w_fea};
    Shape sh_data{num, channel, height, width};
    TensorDf4 tfea(sh_fea);
    TensorDf4 tdata(sh_data);

    TensorDf4 tout;

    LOG(INFO) << "create priorbox param";
    std::vector<float> min_size{60.f};
    std::vector<float> max_size;
    std::vector<float> aspect_ratio{2};
    std::vector<float> variance{0.1f, 0.1f, 0.2f, 0.2f};
    bool flip = true;
    bool clip = false;
    float step_h = 0;
    float step_w = 0;
    int img_w = 0;
    int img_h = 0;
    float offset = 0.5;
    std::vector<PriorType> order;

    order.push_back(PRIOR_MIN);
    order.push_back(PRIOR_MAX);
    order.push_back(PRIOR_COM);

    //PriorBoxParam<TensorDf4> param(min_size, max_size, aspect_ratio, \
                                   variance, flip, clip, img_w, img_h, step_w, step_h, offset, order);
    PriorBoxParam<TensorDf4> param(variance, flip, clip, img_w, img_h, step_w, step_h, offset, order, \
                                    min_size, max_size, aspect_ratio, std::vector<float>(), std::vector<float>(), std::vector<float>());


    std::vector<TensorDf4*> vin;
    std::vector<TensorDf4*> vout;

    vin.push_back(&tfea);
    vin.push_back(&tdata);
    vout.push_back(&tout);
    //print_tensor_device(tdin);
    //cudaDeviceSynchronize();
    //CUDA_POST_KERNEL_CHECK;
    //! create process contex
    Context<NV> ctx_dev(0, 1, 1);

    //! create normalize class
    PriorBox<NV, AK_FLOAT> priorbox;
    LOG(INFO) << "priorbox compute ouput shape";
    SABER_CHECK(priorbox.compute_output_shape(vin, vout, param));
    //LOG(INFO) << "re-alloc tensor buffer";
    Shape va_sh = vout[0]->valid_shape();
    LOG(INFO) << "shape out: " << va_sh[0] << ", " << va_sh[1] << ", " \
              << va_sh[2] << ", " << va_sh[3];
    Shape shape_out{1, 1, 2, w_fea* h_fea * 4 * param.prior_num};
    CHECK_EQ(va_sh == shape_out, true) << "compute output shape error";
    tout.re_alloc(shape_out);

    LOG(INFO) << "priorbox initialization";
    SABER_CHECK(priorbox.init(vin, vout, param, SPECIFY, SABER_IMPL, ctx_dev));

    LOG(INFO) << "priorbox compute";
    SaberTimer<NV> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        SABER_CHECK(priorbox(vin, vout, param, ctx_dev));
        vout[0]->record_event(ctx_dev.get_compute_stream());
        vout[0]->sync();
    }

    CUDA_POST_KERNEL_CHECK;
    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    LOG(INFO) << "total time: " << ts << ", avg time: " << ts / test_iter;
    //print_tensor_device(*output_dev_4d[0]);
    //cudaDeviceSynchronize();
    //Shape sh_show{1, 1, 2, va_sh[2]};
    //TensorHf4 th1(sh_show);
    //th1.copy_from(tout);
    //print_tensor_host(th1);
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

