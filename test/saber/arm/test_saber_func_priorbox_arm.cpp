#include "core/context.h"
#include "funcs/priorbox.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"

using namespace anakin::saber;

int cluster = 0;
int threads = 4;

#define USE_COMPARE
const bool FLAG_RELU = false;

typedef TargetWrapper<ARM> ARM_API;
typedef Tensor<ARM, AK_FLOAT, NCHW> TensorHf4;
template <typename Tensor_t>
void tensor_diff(Tensor_t& t1, Tensor_t& t2, Tensor_t& tdiff) {

    typedef typename Tensor_t::Dtype dtype;
    int size1 = t1.valid_size();
    int size2 = t2.valid_size();
    int size_out = tdiff.valid_size();
            CHECK_EQ(size1, size2) << "wrong shape";
            CHECK_EQ(size1, size_out) << "wrong shape";
    const dtype* ptr1 = t1.data();
    const dtype* ptr2 = t2.data();
    dtype* ptr_out = tdiff.mutable_data();
    for (int i = 0; i < size1; ++i) {
        ptr_out[i] = ptr1[i] - ptr2[i];
    }
}

void test_arm_priorbox(std::vector<TensorHf4*>& tin, \
    int thread_num, int cluster_id) {

    int test_iter = 100;
    double to = 0;
    double min_time = 1000000;
    SaberTimer<ARM> t1;

    Context<ARM> ctx1;
    PowerMode mode = cluster_id == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
            LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
                LOG(INFO) << "number of threads: " << thread;
#endif
    }

    TensorHf4 tout_saber;
    std::vector<TensorHf4*> tvout_saber;
    tvout_saber.push_back(&tout_saber);

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

    PriorBoxParam<TensorHf4> param(min_size, max_size, aspect_ratio, \
        variance, flip, clip, img_w, img_h, step_w, step_h, offset, order);
    PriorBox<ARM, AK_FLOAT> priorbox_saber;

    priorbox_saber.compute_output_shape(tin, tvout_saber, param);
    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    Shape shape_out{1, 1, 2, tin[0]->width() * tin[0]->height() * 4 * param.prior_num};

    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

    LOG(INFO) << "saber priorbox impl init";
    SABER_CHECK(priorbox_saber.init(tin, tvout_saber, param, SPECIFY, SABER_IMPL, ctx1));

    //! compute
    LOG(INFO) << "saber priorbox compute";
    to = 0;

    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        priorbox_saber(tin, tvout_saber, param, ctx1);
        tvout_saber[0]->record_event(ctx1.get_compute_stream());
        tvout_saber[0]->sync();
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "saber conv running time, ave: " << to / test_iter << ", min time: " << min_time;
    print_tensor_host(*tvout_saber[0]);

}

TEST(TestSaberFuncTest, test_func_priorbox_arm) {

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
    TensorHf4 tfea(sh_fea);
    TensorHf4 tdata(sh_data);

    std::vector<TensorHf4*> tin;

    tin.push_back(&tfea);
    tin.push_back(&tdata);

    test_arm_priorbox(tin, threads, cluster);
}

int main(int argc, const char** argv){

    Env<ARM>::env_init();

    // initial logger
    //logger::init(argv[0]);

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

