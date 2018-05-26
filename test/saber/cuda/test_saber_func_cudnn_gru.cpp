#include <vector>
#include "core/context.h"
#include "test_saber_func_NV.h"
#include "tensor_op.h"
#include "funcs/impl/cuda/cudnn_helper.h"
#include "funcs/gru.h"
#include "saber_types.h"
#include "saber/funcs/timer.h"

#include "stdio.h"


using namespace anakin::saber;


void write_tensorfile(Tensor<X86, AK_FLOAT, NCHW> tensor, const char *locate) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW>::Dtype Dtype;
    LOG(INFO) << "host tensor data:" << tensor.size();
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        LOG(ERROR) << "file open field " << locate;

    } else {
        const Dtype* data_ptr = static_cast<const Dtype*>(tensor.data());
        int size = tensor.valid_size();

        for (int i = 0; i < size; ++i) {
            fprintf(fp, "[%d] %f \n", i,(data_ptr[i]));

        }

        fclose(fp);
    }

    LOG(INFO) << "!!! write success: " << locate;
}

void readTensorData(Tensor<X86, AK_FLOAT, NCHW> tensor, const char* locate) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW>::Dtype Dtype;
    FILE* fp = fopen(locate, "rb");

    if (fp == 0) {
        LOG(ERROR) << "file open failed " << locate;

    } else {
        LOG(INFO) << "file open success [" << locate << " ],read " << tensor.valid_shape().count();
        fread(tensor.mutable_data(), sizeof(Dtype), tensor.valid_size(), fp);
        fclose(fp);
    }
}

//#define printTensorShape(tensor)\
//do{\
//LOG(INFO)<<"("<<tensor.num()<<","<<tensor.channel()<<","<<tensor.height()<<","<<tensor.width()<<")";\
//}while(0)
//
#define printShape(tensor)\
do{\
LOG(INFO)<<"("<<tensor[0]<<","<<tensor[1]<<","<<tensor[2]<<","<<tensor[3]<<")";\
}while(0)
//#define GRUOFFSET
void test_cudnn_gru() {
    Context<NV> ctx_dev(0, 1, 1);
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;
    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;



    int img_num = 1;
    int in_channels = 5;
    int img_h = 5; //231;
    int img_w = 3; //727;
    int hidden_size =4;

    int batch_size=img_h;
    int sequence_size=in_channels;
    int word_size=img_w;
//fill_tensor_host_seq()
    TensorHf4 img_host_in;
    TensorHf4 img_host_out;
    TensorHf4 img_host_from_dev;
    TensorHf4 img_host_to_dev;
    TensorHf4 img_host_hidden;

    TensorDf4 img_dev_in;
    TensorDf4 img_dev_hidden;
    TensorDf4 img_dev_out;


    TensorHf4 host_weights;
    TensorHf4 host_bias;
    TensorDf4 dev_weights;
    TensorDf4 dev_bias;

#ifdef GRUOFFSET
    Shape img_s_shape(1, 1, sequence_size*15, word_size);
    //TODO:test sequence,batch
    Shape img_out_shape(1, 1, sequence_size*15, hidden_size);
#else
    Shape img_s_shape(1, sequence_size, batch_size, word_size);
    Shape img_out_shape(1, sequence_size, batch_size, hidden_size);
#endif

    Shape img_weights_shape(1, 1, 1, hidden_size * hidden_size * 3 + word_size * hidden_size * 3);
    Shape img_bias_shape(1, 1, 1, 3 * hidden_size);
    Shape img_hidden_shape(1, 1, batch_size, hidden_size);


    img_host_in.re_alloc(img_s_shape);
    img_dev_in.re_alloc(img_s_shape);
    img_host_to_dev.re_alloc(img_s_shape);

    img_host_out.re_alloc(img_out_shape);
    img_dev_out.re_alloc(img_out_shape);
    img_host_from_dev.re_alloc(img_out_shape);

    host_weights.re_alloc(img_weights_shape);
    dev_weights.re_alloc(img_weights_shape);

    host_bias.re_alloc(img_bias_shape);
    dev_bias.re_alloc(img_bias_shape);

    img_host_hidden.re_alloc(img_hidden_shape);
    img_dev_hidden.re_alloc(img_hidden_shape);

    //    fill_tensor_host_seq(img_host_in);
    readTensorData(img_host_in, "x_in");
    img_dev_in.copy_from(img_host_in);
#ifdef GRUOFFSET
    std::vector<int> offsets={0,sequence_size,sequence_size*3,sequence_size*6,sequence_size*10,sequence_size*15};
    img_dev_in.set_seq_offset(offsets);
#else

#endif
    readTensorData(host_weights, "UW");
    dev_weights.copy_from(host_weights);

    readTensorData(host_bias, "bias");
    dev_bias.copy_from(host_bias);

    readTensorData(img_host_hidden, "h_in");
    img_dev_hidden.copy_from(img_host_hidden);

    readTensorData(img_host_out, "y_out");
    write_tensorfile(img_host_out, "host_in_y.txt");

    LOG(INFO) << "before real run: ";
    //    compute_transpose_gold(img_host_out.mutable_data(), img_host_in.data(),
    //                         img_num, in_channels, img_h, img_w);

    //    img_host_to_dev.copy_from(img_dev_in);

    std::vector<TensorDf4*> input_dev_4d;
    std::vector<TensorDf4*> output_dev_4d;
    Gru<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> dev_gru;
    input_dev_4d.push_back(&img_dev_in);
    input_dev_4d.push_back(&img_dev_hidden);
    output_dev_4d.push_back(&img_dev_out);
#ifdef GRUOFFSET
    GruParam<TensorDf4> param(&dev_weights, &dev_bias, GRU_CUDNN,1.0, 1, 1, true);
#else
    GruParam<TensorDf4> param(&dev_weights, &dev_bias, GRU_CUDNN,1.0, 1, 1, false);
#endif


    dev_gru.compute_output_shape(input_dev_4d, output_dev_4d, param);
    LOG(INFO) << "shape of output =" << img_dev_out.valid_shape().data()[0] << "," <<
              img_dev_out.valid_shape().data()[1] << "," << img_dev_out.valid_shape().data()[2] << "," <<
              img_dev_out.valid_shape().data()[3];
    output_dev_4d[0]->re_alloc(output_dev_4d[0]->valid_shape());
    //    printTensorShape(img_dev_out);
    Shape shape = output_dev_4d[0]->get_stride();
    LOG(INFO) << "(" << shape[0];
    printShape(shape);


    LOG(INFO) << "dev_gru initialization";
    SABER_CHECK(dev_gru.init(input_dev_4d, output_dev_4d, param, SPECIFY, VENDER_IMPL, ctx_dev));


    int test_iter = 1;
    SaberTimer<NV> t1;
    t1.start(ctx_dev);

    for (int i = 0; i < test_iter; ++i) {
        dev_gru(input_dev_4d, output_dev_4d, param, ctx_dev);
        output_dev_4d[0]->record_event(ctx_dev.get_compute_stream());
        output_dev_4d[0]->sync();
    }

    t1.end(ctx_dev);

    LOG(INFO) << test_iter << "test, total time: " << t1.get_average_ms() << "avg time : " << \
              t1.get_average_ms() / test_iter;
#if 1
    img_host_from_dev.copy_from(img_dev_out);
    double maxratio = 0;
    double maxdiff = 0;
    tensor_cmp_host(img_host_out.data(), img_host_from_dev.data(), img_host_from_dev.size(), maxratio,
                    maxdiff);
    write_tensorfile(img_host_out, "host.txt");
//            LOG(INFO) << "before host ";
    write_tensorfile(img_host_from_dev, "dev.txt");
    if (maxdiff > 0.0001) {
        LOG(INFO) << "before print: ";
        write_tensorfile(img_host_out, "host.txt");
        LOG(INFO) << "before host ";
        write_tensorfile(img_host_from_dev, "dev.txt");
        LOG(ERROR) << img_num << "," << in_channels << "," << img_h << "," << img_w;
        return;
    }
#endif
    LOG(INFO) << "passed";
    return;

    //    printShape(img_host_in);
    //    printShape(img_dev_out);
    //            LOG(INFO) << img_num<<","<<in_channels<<","<<img_h<<","<<img_w<<","<<maxdiff<<","<<maxratio;
    //    if(maxdiff>0.01){
    //                LOG(INFO) << "before print: ";
    //        write_tensorfile(img_host_out,"host.txt");
    //                LOG(INFO) << "before host ";
    //        write_tensorfile(img_host_from_dev,"dev.txt");
    //        write_tensorfile(img_host_to_dev,"origin.txt");
    //                LOG(ERROR) << img_num<<","<<in_channels<<","<<img_h<<","<<img_w;
    //        return;
    //    }
    //    //compute_transpose_gold(img_host_out.mutable_data(), static_cast<const float *>(img_dev_in.get_buf()->get_data()),img_num,in_channels,img_h,img_w);
    ////    }
    //            LOG(INFO) << "passed";
}

TEST(TestSaberFuncNV, test_func_transpose) {

    typedef Tensor<X86, AK_FLOAT, NCHW> TensorHf4;
    typedef Tensor<NV, AK_FLOAT, NCHW> TensorDf4;

    typedef Tensor<X86, AK_INT8, NCHW> TensorHINT8;
    typedef Tensor<NV, AK_INT8, NCHW> TensorDINT8;

    test_cudnn_gru();

    //    test_cudnn_tensor<TensorHINT8, TensorDINT8, TensorHINT8, TensorDINT8>();

    //    test_cudnn_tensor<TensorHINT8, TensorDINT8>();

}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    Env<NV>::env_init();

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
