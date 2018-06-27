#include "core/context.h"
#include "funcs/softmax.h"
#include "test_saber_func_softmax_BM.h"
#include "tensor_op.h"
#include "saber_types.h"
#include <vector>

using namespace anakin::saber;

TEST(TestSaberFuncSoftmaxBM, test_func_softmax_BM) {

    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;

    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    typedef TensorDf4::Dtype dtype;

    int test_iter = 1000;

    int softmax_axis = 3; // channel
    int w_in = 3;
    int h_in = 225;
    int ch_in = 40;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_out = shape_in;

    SoftmaxParam<TensorDf4> param(softmax_axis);

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    LOG(INFO) << "softmax axis= " << param.axis;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);

    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = i % 4;
    }

    TensorDf4 tdin, tdout;
    tdin.re_alloc(shape_in);
    tdin.copy_from(thin);
    input_dev_4d.push_back(&tdin);

    // start Reshape & doInfer
    Context<BM> ctx_dev(0, 1, 1);

    Softmax<BM, AK_BM> softmax_dev;

    typedef std::vector<Shape> Shape_v;

    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
              shape_out[2] << ", " << shape_out[3];

    output_dev_4d.push_back(&tdout);
    softmax_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);

    LOG(INFO) << "re-alloc tensor buffer";
    output_dev_4d[0]->re_alloc(output_dev_4d[0]->shape());

    LOG(INFO) << "softmax initialized to cudnn impl";
    softmax_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev);

    LOG(INFO) << "cudnn softmax compute";
    SaberTimer<BM> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        softmax_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    printf("cudnn softmax total time : %.4f, avg time : %.4f\n", ts, ts / test_iter);

    LOG(INFO) << "softmax initialized to saber impl";
    softmax_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev);

    LOG(INFO) << "saber softmax compute";
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        softmax_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);
    ts = t1.get_average_ms();
    printf("saber softmax total time : %.4f, avg time : %.4f\n", ts, ts / test_iter);
    //print_tensor_device(*output_dev_4d[0]);
}

TEST(TestSaberFuncSoftmaxBM, test_func_softmax_ROI_BM) {

    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;

    typedef Tensor<BM, AK_BM, NCHW> TensorDf4;

    typedef TensorDf4::Dtype dtype;

    int test_iter = 1;

    int softmax_axis = 3; // channel
    int w_in = 3;
    int h_in = 10;
    int ch_in = 10;
    int num_in = 1;

    Shape shape_in(num_in, ch_in, h_in, w_in);
    Shape shape_in_roi{num_in, ch_in / 2, h_in / 2, w_in};
    Shape shape_out = shape_in_roi;

    SoftmaxParam<TensorDf4> param(softmax_axis);

    LOG(INFO) << " input tensor size, num=" << num_in << ", channel=" << \
              ch_in << ", height=" << h_in << ", width=" << w_in;

    LOG(INFO) << "softmax axis= " << param.axis;

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;

    Tensor<X86, AK_FLOAT, NCHW> thin(shape_in);

    for (int i = 0; i < thin.size(); ++i) {
        thin.mutable_data()[i] = (i % 3);
    }

    TensorDf4 tdin, tdin_roi, tdout, tdout_roi;
    tdin.re_alloc(shape_in);
    tdout.re_alloc(shape_in);
    tdin.copy_from(thin);
    tdin_roi.share_sub_buffer(tdin, shape_in_roi, Shape(0, 0, 0, 0));
    input_dev_4d.push_back(&tdin_roi);
    output_dev_4d.push_back(&tdout_roi);

    // start Reshape & doInfer
    Context<BM> ctx_dev(0, 1, 1);

    Softmax<BM, AK_BM> softmax_dev;

    LOG(INFO) << "shape out 4d: " << shape_out[0] << ", " << shape_out[1] << ", " << \
              shape_out[2] << ", " << shape_out[3];

    softmax_dev.compute_output_shape(input_dev_4d, output_dev_4d, param);

    LOG(INFO) << "re-alloc tensor buffer";
    output_dev_4d[0]->share_sub_buffer(tdout, shape_in_roi, Shape(0, 0, 0, 0));
    //output_dev_4d[0]->reshape(output_dev_4d[0]->valid_shape());

    LOG(INFO) << "softmax initialization";
    softmax_dev.init(input_dev_4d, output_dev_4d, param, SPECIFY, SABER_IMPL, ctx_dev);

    LOG(INFO) << "softmax compute";
    SaberTimer<BM> t1;
    t1.clear();
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        softmax_dev(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);
    float ts = t1.get_average_ms();
    printf("total time : %.4f, avg time : %.4f\n", ts, ts / test_iter);
    print_tensor_device(*output_dev_4d[0]);

    TensorDf4 troi(output_dev_4d[0]->valid_shape());
    troi.copy_from(*output_dev_4d[0]);
    print_tensor_device(troi);
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

