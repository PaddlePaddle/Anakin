#include "graph_base.h"
#include "graph.h"
#include "scheduler.h"
#include "net.h"
#include "worker.h"
#include "tensor_op.h"
#include "timer.h"
#include "saber/utils.h"

using namespace anakin::saber;
using namespace anakin::graph;
using namespace anakin;
typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4hf;
typedef Tensor<AMD, AK_FLOAT, NCHW> Tensor4df;

void load_labels(std::string path, std::vector<std::string>& labels) {

    FILE* fp = fopen(path.c_str(), "r");
    if (fp == nullptr) {
        LOG(FATAL) << "load label file failed";
    }
    while (!feof(fp)) {
        char str[1024];
        fgets(str, 1024, fp);
        std::string str_s(str);

        if (str_s.length() > 0) {
            for (int i = 0; i < str_s.length(); i++) {
                if (str_s[i] == ' ') {
                    std::string strr = str_s.substr(i, str_s.length() - i - 1);
                    labels.push_back(strr);
                    i = str_s.length();
                }
            }
        }
    }
    fclose(fp);
}

void print_topk(const float* scores, const int size, const int topk, \
    const std::vector<std::string>& labels) {

    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++) {
        float score = vec[i].first;
        int index = vec[i].second;
        LOG(INFO) << i <<": " << index << "  " << labels[index] << "  " << score;
    }
}

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"

using namespace cv;

void fill_tensor_with_cvmat(const Mat& img_in, Tensor4hf& tout, const int num, \
    const int width, const int height, const float* mean, const float* scale) {
    cv::Mat im;
    cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
    float* ptr_data_in = tout.mutable_data();
    int stride = width * height;
    for (int i = 0; i < num; i++) {
        float* ptr_in = ptr_data_in + i * tout.channel() * tout.height() * tout.width();
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                ptr_in[r * width + c] = (im.at<cv::Vec3b>(r, c)[0] - mean[0]) * scale[0];
                ptr_in[stride + r * width + c] = (im.at<cv::Vec3b>(r, c)[1] - mean[1]) * scale[1];
                ptr_in[2 * stride + r * width + c] = (im.at<cv::Vec3b>(r, c)[2] - mean[2]) * scale[2];
            }
        }
    }
}
#endif

void test_net(const std::string model_file_name, const std::string image_file_name, \
    const std::vector<std::string>& labels, const int topk, const int threads, \
    const int test_iter) {

    int batch_size = 1;

    //! create runtime context
    LOG(INFO) << "create runtime context";
    std::shared_ptr<Context<AMD>> ctx1 = std::make_shared<Context<AMD>>(0,0,0);

    //! load model
    LOG(WARNING) << "load anakin model file from " << model_file_name << " ...";
    Graph<AMD, AK_FLOAT, Precision::FP32> graph;
    auto status = graph.load(model_file_name);
    if (!status) {
         LOG(FATAL) << " [ERROR] " << status.info();
    }

    //! set batch size
    graph.ResetBatchSize("input_0", batch_size);

    //! optimize the graph
    LOG(INFO) << "optimize the graph";
    graph.Optimize();

    //! get output name
    std::vector<std::string>& vout_name = graph.get_outs();
    LOG(INFO) << "output size: " << vout_name.size();

    //! constructs the executer net
    LOG(INFO) << "create net to execute";
    Net<AMD, AK_FLOAT, Precision::FP32, OpRunType::SYNC> net_executer(graph, ctx1, true);

    //! get in
    LOG(INFO) << "get input";
    auto d_tensor_in_p = net_executer.get_in("input_0");
    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i = 0; i < valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
    }
    Tensor4hf thin(valid_shape_in);

    LOG(INFO) << thin.width() << "x" << thin.height() << " size" << thin.valid_size();;
    //! feed input image to input tensor

#ifdef USE_OPENCV
    LOG(INFO) << "loading image " << image_file_name << " ...";
    Mat img = imread(image_file_name, CV_LOAD_IMAGE_COLOR);
    if (img.empty()) {
        LOG(FATAL) << "opencv read image " << image_file_name << " failed";
    }
    //! set your mean value and scale value here
    float mean_mb[3] = {103.94f, 116.78f, 123.68f};
    float scale_mb[3] = {0.017f, 0.017f, 0.017f};
    LOG(INFO) << thin.width() << "x" << thin.height();
    fill_tensor_with_cvmat(img, thin, batch_size, thin.width(), thin.height(), mean_mb, scale_mb);
#else
    fill_tensor_host_const(thin, 1.f);
#endif

    //! do inference
    Context<AMD> ctx(0, 0, 0);
    anakin::saber::SaberTimer<AMD> my_time;
    LOG(INFO) << "run prediction ";

    double to = 0;
    double tmin = 1000000;
    double tmax = 0;
    my_time.start(ctx);
    saber::SaberTimer<AMD> t1;
    for (int i = 0; i < test_iter; i++) {
        d_tensor_in_p->copy_from(thin);
        t1.clear();
        t1.start(ctx);
        net_executer.prediction();
        t1.end(ctx);
        double tdiff = t1.get_average_ms();
        if (tdiff > tmax) {
            tmax = tdiff;
        }
        if (tdiff < tmin) {
            tmin = tdiff;
        }
        to += tdiff;
    }
    my_time.end(ctx);


    LOG(INFO) << model_file_name << " batch_size " << batch_size << \
        " average time " << to / test_iter << \
        ", min time: " << tmin << "ms, max time: " << tmax << " ms";

    //! get output
    //! fixme get output
    //std::vector<Tensor4hf*> vout = net_executer.get_out_list();
    std::vector<Tensor4df*> vout;
    for (auto& it : vout_name) {
        vout.push_back(net_executer.get_out(it));
    }
    Tensor4df* tensor_out_d = vout[0];
    LOG(INFO) << "output size: " << vout.size();

    Tensor4hf tensor_out;
    tensor_out.re_alloc(tensor_out_d->shape());
    tensor_out.copy_from(*tensor_out_d);
#if 0 //print output tensor data
    LOG(INFO) << "extract data: size: " << tensor_out->valid_size() << \
        ", width=" << tensor_out->width() << ", height=" << tensor_out->height();
    const float* ptr_out = tensor_out->data();
    for (int i = 0; i < tensor_out->valid_size(); i++) {
        printf("%0.4f  ", ptr_out[i]);
        if ((i + 1) % 7 == 0) {
            printf("\n");
        }
    }
    printf("\n");
#endif
    print_topk(tensor_out.data(), tensor_out.valid_size(), topk, labels);
}

int main(int argc, char** argv){

    LOG(INFO) << "initialized the device";
    Env<AMD>::env_init();

    if (argc < 4) {
        LOG(ERROR) << "usage: " << argv[0] << ": model_file label_file image_name [topk] [test_iter] [threads]";
        return -1;
    }
    char* model_file = argv[1];
    char* label_file = argv[2];
    char* image_path = argv[3];

    std::vector<std::string> labels;
    load_labels(label_file, labels);

    int topk = 5;
    if (argc > 4) {
        topk = atoi(argv[4]);
    }

    int test_iter = 10;
    if (argc > 5) {
        test_iter = atoi(argv[5]);
    }

    int threads = 1;
    if (argc > 6) {
        threads = atoi(argv[6]);
    }

	test_net(model_file, image_path, labels, topk, threads, test_iter);
    return 0;
}

