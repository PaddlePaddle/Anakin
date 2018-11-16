#include "saber/lite/funcs/saber_concat.h"
#include "test_lite.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

int cluster = 0;
int threads = 4;

typedef Tensor<CPU> TensorH;

template <typename dtype>
void concat_basic(const std::vector<TensorH*>& inputs, std::vector<TensorH*>& outputs, ConcatParam& param){

    int axis = param._axis;
    int num = outputs[0]->num();
    int channel = outputs[0]->channel();
    int height = outputs[0]->height();
    int width = outputs[0]->width();

    Shape out_sh = outputs[0]->valid_shape();
    int out_concat_axis = out_sh[axis];
    int num_concats = inputs[0]->count_valid(0, param._axis);
    int concat_input_size = inputs[0]->count_valid(param._axis + 1, inputs[0]->dims());

    dtype* dout = (dtype*)outputs[0]->mutable_data();
    int total_size = out_concat_axis * concat_input_size;

    for(int k = 0; k < num_concats; k++){
        dtype* dout_ptr = dout + k * total_size;
        int out_size = 0;
        for(int i = 0; i < inputs.size(); i++){
            Shape in_sh = inputs[i]->valid_shape();
            int size = in_sh[axis] * concat_input_size;
            const dtype* din = (dtype*)inputs[i]->data();
            const dtype* din_ptr = din + k * size;
            dtype* dout_ptr_axis = dout_ptr + out_size;
            for(int j = 0; j < size; j++){
                dout_ptr_axis[j] = din_ptr[j];
            }
            out_size += size;
        }
    }
}

TEST(TestSaberLite, test_func_concat_arm) {

    Context ctx1;
    PowerMode mode = SABER_POWER_HIGH;
    ctx1.set_run_mode(mode, threads);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    const int test_iter = 100;

    SaberConcat concat_lite;
    for (auto& axis : {0, 1, 2, 3}) {
        ConcatParam param(axis);
        concat_lite.load_param(&param);
        for (auto& type : {AK_FLOAT, AK_INT8}) {
            int n = get_rand(1, 10);
            int c = get_rand(1, 100);
            int h = get_rand(1, 100);
            int w = get_rand(1, 100);

            Shape sh1 = {n, c, h, w};
            Shape sh2 = sh1;
            Shape sh3 = sh1;
            sh1[axis] = get_rand(1, 100);
            sh2[axis] = get_rand(1, 100);
            sh3[axis] = get_rand(1, 100);

            Shape shape_out = sh1;
            shape_out[axis] = sh1[axis] + sh2[axis] + sh3[axis];
            LOG(INFO) << " input size, num=" << n << ", channel=" << \
                c << ", height=" << h << ", width=" << w;
            LOG(INFO) << "concat axis= " << axis << ", size: " << sh1[axis] << \
                ", " << sh2[axis] << ", " << sh3[axis];
            LOG(INFO) << "compute precision: " << ((type == AK_FLOAT)? "float" : "int8");

            //! prepare inputs and outputs
            std::vector<TensorH*> vin;
            std::vector<TensorH*> vout;

            TensorH th1, th2, th3;
            th1.re_alloc(sh1, type);
            th2.re_alloc(sh2, type);
            th3.re_alloc(sh3, type);
            fill_tensor_rand(th1, -100, 100);
            fill_tensor_rand(th2, -100, 100);
            fill_tensor_rand(th3, -100, 100);
            vin.push_back(&th1);
            vin.push_back(&th2);
            vin.push_back(&th3);

            TensorH tdev_out;
            vout.push_back(&tdev_out);

            concat_lite.compute_output_shape(vin, vout);
            LOG(INFO) << "output shape: " << tdev_out.valid_shape()[0] << ", " \
              << tdev_out.valid_shape()[1] << ", " << tdev_out.valid_shape()[2] \
              << ", " << tdev_out.valid_shape()[3];

            CHECK_EQ(shape_out == vout[0]->valid_shape(), true) << "compute shape error";
            tdev_out.re_alloc(shape_out, type);

            //! set op precision type
            concat_lite.set_op_precision(type);

            concat_lite.init(vin, vout, ctx1);

            SaberTimer t1;
            t1.clear();
            t1.start();

            for (int i = 0; i < test_iter; ++i) {
                concat_lite.dispatch(vin, vout);
            }

            t1.end();
            float ts = t1.get_average_ms();
            LOG(INFO) << "total time : " << ts << ", avg time : " << ts / test_iter;

            std::vector<TensorH*> vout_basic;
            TensorH tout_basic;
            tout_basic.re_alloc(shape_out, type);
            vout_basic.push_back(&tout_basic);

            if (type == AK_FLOAT) {
                concat_basic<float>(vin, vout_basic, param);
            } else if (type == AK_INT8) {
                concat_basic<char>(vin, vout_basic, param);
            } else {
                LOG(FATAL) << "unsupported dtype";
            }

            double max_ratio;
            double max_diff;
            tensor_cmp_host(*vout[0], *vout_basic[0], max_ratio, max_diff);
            CHECK_EQ(fabsf(max_ratio) < 1e-6f, true) << "concat compute result error";
            LOG(INFO) << "finished compare, pass!";
        }
    }
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env::env_init(4);

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

