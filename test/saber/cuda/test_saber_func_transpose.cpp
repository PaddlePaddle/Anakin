#include <vector>
#include "core/context.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/impl/cuda/cudnn_helper.h"
#include "funcs/transpose.h"
#include "saber_types.h"
#include "saber/funcs/timer.h"
#include "stdio.h"

using namespace anakin::saber;

static void write_tensorfile(Tensor <X86, AK_FLOAT, NCHW> tensor, const char* locate) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW>::Dtype Dtype;
    LOG(INFO) << "host tensor data:" << tensor.size();
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        LOG(ERROR) << "file open failed " << locate;

    } else {
        const Dtype* data_ptr = static_cast<const Dtype*>(tensor.data());
        int size = tensor.size();

        for (int i = 0; i < size; ++i) {
            fprintf(fp, "%8.0f ", static_cast<float>(data_ptr[i]));

            if ((i + 1) % tensor.width() == 0) {
                fprintf(fp, "\n");
            }
        }

        fclose(fp);
    }
}

static void compute_transpose_gold(float* gold, const float* idata, const int num,
                                   const int channel,
                                   const int height, const int width) {
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channel; ++c) {
            int offset = n * channel * height * width + c * height * width;

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    gold[(x * height) + y + offset] = idata[(y * width) + x + offset];
                }
            }
        }
    }
}

#define printShape(tensor)                      \
    do{                                         \
        LOG(INFO) << "(" << tensor.num() << "," \
                  << tensor.channel() << ","    \
                  << tensor.height() << ","     \
                  << tensor.width() << ")";     \
    }while(0)

void test_transpose() {
    Context<NV> ctx_dev(0, 1, 1);
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;

    int img_num = 2;
    int in_channels = 33;
    int img_h = 231;
    int img_w = 727; 
    TensorHf4 img_host_in;
    TensorHf4 img_host_out;
    TensorHf4 img_host_from_dev;
    TensorHf4 img_host_to_dev;

    TensorDf4 img_dev_in;
    TensorDf4 img_dev_out;

    Shape img_s(img_num, in_channels, img_h, img_w);
    Shape img_out(img_num, in_channels, img_w, img_h);

    img_host_in.re_alloc(img_s);
    img_dev_in.re_alloc(img_s);
    img_host_to_dev.re_alloc(img_s);

    img_host_out.re_alloc(img_out);
    img_dev_out.re_alloc(img_out);
    img_host_from_dev.re_alloc(img_out);

    fill_tensor_host_seq(img_host_in);
    img_dev_in.copy_from(img_host_in);

    LOG(INFO) << "Before real run: ";
    compute_transpose_gold(img_host_out.mutable_data(), img_host_in.data(),
                           img_num, in_channels, img_h, img_w);

    img_host_to_dev.copy_from(img_dev_in);

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    Transpose<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> dev_transpose;
    input_dev_4d.push_back(&img_dev_in);
    output_dev_4d.push_back(&img_dev_out);

    TransposeParam<TensorDf4> param;

    dev_transpose.compute_output_shape(input_dev_4d, output_dev_4d, param);
    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());

    LOG(INFO) << "Transpose initialization";
    SABER_CHECK(dev_transpose.init(input_dev_4d, output_dev_4d, param\
            , RUNTIME, SABER_IMPL, ctx_dev));

    int test_iter = 11;
    SaberTimer<NV> t1;
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        dev_transpose(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);

    LOG(INFO) <<"Iterator time: " << test_iter << " total time: " << t1.get_average_ms() ;
    LOG(INFO) << "Avg time : " << t1.get_average_ms() / test_iter;

    img_host_from_dev.copy_from(img_dev_out);
    double maxratio = 0;
    double maxdiff = 0;
    tensor_cmp_host(img_host_out.data(), img_host_from_dev.data(),
                    img_host_from_dev.size(), maxratio, maxdiff);
    LOG(WARNING) << " Before shape: ";
    printShape(img_host_in);
    LOG(WARNING) << " After shape: ";
    printShape(img_dev_out);

    if (maxdiff > 0.01) {
        LOG(INFO) << "before print: ";
        write_tensorfile(img_host_out, "host.txt");
        LOG(INFO) << "before host ";
        write_tensorfile(img_host_from_dev, "dev.txt");
        write_tensorfile(img_host_to_dev, "origin.txt");
        LOG(ERROR) << img_num << "," << in_channels << "," << img_h << "," << img_w;
        return;
    }

    LOG(INFO) << "Passed";
}

TEST(TestSaberFuncNV, test_func_transpose) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef Tensor<X86, AK_INT8, NCHW> TensorHINT8;
    typedef Tensor<NV, AK_INT8, NCHW> TensorDINT8;

    test_transpose();
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    Env<NV>::env_init();

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
