#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "debug.h"
#ifdef ENABLE_OP_TIMER
#include"saber/funcs/impl/impl_base.h"
#endif
#ifdef USE_ARM_PLACE
#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
using namespace cv;
#endif

std::string g_model_path = "";
int g_batch_size = 1;
std::string g_img_path = "val_list.txt";
std::string g_img_file = "/data/local/tmp";
int g_thread_num = 1;
int g_cluster = 0;
bool g_set_archs = false;
ARMArch g_arch = A73;

#ifdef USE_OPENCV
static void fill_tensor_with_cvmat(const Mat& im, float* dout, const int num, const int channel, \
    const int width, const int height, const float* mean, const float* scale) {
    int stride = width * height;
    for (int i = 0; i < num; i++) {
        float* ptr_out = dout + i * channel * height * width;
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                ptr_out[r * width + c] = (im.at<cv::Vec3f>(r, c)[2] - mean[0]) * scale[0];
                ptr_out[stride + r * width + c] = (im.at<cv::Vec3f>(r, c)[1] - mean[1]) * scale[1];
                ptr_out[2 * stride + r * width + c] = (im.at<cv::Vec3f>(r, c)[0] - mean[2]) * scale[2];
            }
        }
    }
}

int calc_top1(float* data, int size, int label){
    float max = -1.f;
    int max_idx = -1;
    for (int i = 0; i < size; ++i){
        if (data[i] > max){
            max = data[i];
            max_idx = i;
        }
    }
    return int(max_idx == label);
}

int calc_top5(float* data, int size, int label){
    float max = -1.f;
    int max_idx = -1;
    bool flag = false;
    for (int k = 0; k < 5; ++k) {
        for (int i = 0; i < size; ++i) {
            if (data[i] > max) {
                max = data[i];
                max_idx = i;
            }
        }
        flag = flag || (max_idx == label);
        data[max_idx] = -1.f;
        max = -1.f;
    }
    return int(flag);
}

Mat pre_process_img(Mat& im, int width, int height){
    float percent = 256.f / std::min(im.cols, im.rows);
    int resized_width = int(roundf(im.cols * percent));
    int resized_height = int(roundf(im.rows * percent));
    resize(im ,im, Size(resized_width, resized_height), INTER_LANCZOS4);
    int crop_width = width;
    int crop_height = height;
    int w_start = (im.cols - crop_width) / 2;
    int h_start = (im.rows - crop_height) / 2;
    Rect roi;
    roi.x = w_start;
    roi.y = h_start;
    roi.width = crop_width;
    roi.height = crop_height;
    Mat crop = im(roi);
    return crop;
}
//! set your mean value and scale value here
//float mean_mb[3] = {103.939, 116.779, 123.68};
float mean_mb[3] = {0.485, 0.456, 0.406};
//float scale_mb[3] = {1.f, 1.f, 1.f}; // for resnet
float scale_mb[3] = {1.f / 0.229, 1.f / 0.224, 1.f / 0.225}; // mobilenet

TEST(NetTest, net_execute_base_test) {
    LOG(INFO) << "begin test";
    Context<ARM> ctx1;
    ctx1.set_run_mode((PowerMode)g_cluster, g_thread_num);
    if (g_set_archs) {
        ctx1.set_arch(g_arch);
        LOG(INFO) << "arm arc: " << g_arch;
    }
    ctx1.set_cache(32 * 1024, 512* 1024, 0);
#ifdef USE_OPENCV
    using namespace cv;
#endif
    Graph<ARM, Precision::INT8>* graph = new Graph<ARM, Precision::INT8>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);

    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    std::vector<std::string>& vin_name = graph->get_ins();
    LOG(INFO) << "number of input tensor: " << vin_name.size();

    for (int j = 0; j < vin_name.size(); ++j) {
        graph->ResetBatchSize("input_0", g_batch_size);
    }

    graph->Optimize();

    Net<ARM, Precision::INT8> net_executer(true);
    net_executer.init(*graph);

    for (int j = 0; j < vin_name.size(); ++j) {
        Tensor<ARM>* d_tensor_in_p = net_executer.get_in(vin_name[j]);
        Shape shin = d_tensor_in_p->valid_shape();
        //tin->reshape(Shape(1, 3, 224, 224));
        LOG(INFO) << "input tensor size: ";
        //Shape shin = tin->valid_shape();
        LOG(INFO) << "input name: " << vin_name[j];
        for (int k = 0; k < d_tensor_in_p->dims(); ++k) {
            LOG(INFO) << "|---: " << shin[k];
        }
        fill_tensor_const(*d_tensor_in_p, 1.f);
    }
    printf("------------ start to test\n");
    std::vector<std::string>& out_name = graph->get_outs();
    LOG(INFO) << "number of output tensor: " << out_name.size();
    for (int i = 0; i < out_name.size(); i++) {
        Tensor<ARM>* vout = net_executer.get_out(out_name[i]);
        LOG(INFO) << "output tensor size: ";
        Shape shout = vout->valid_shape();
        for (int j = 0; j < vout->dims(); ++j) {
            LOG(INFO) << "|---: " << shout[j];
        }
    }

    LOG(WARNING) << "pre-deal !!!!!!!! ";
    // ==================== precision ===================
    float top1_sum = 0;
    float top5_sum = 0;
    int total_count = 0;
    // ==================================================
    std::vector<std::string> img_list;
    std::vector<int> labels;
    //! load test image list
    std::fstream fp_img(g_img_path);
    std::string line;
    while (getline(fp_img, line)) {
        std::string path = line.substr(0, line.find(" "));
        std::string label = line.substr(line.find(" "));
        path = g_img_file + path;
        LOG(INFO) << "img_file_path: " <<path;
        img_list.push_back(path);
        labels.push_back(atoi(label.c_str()));
    }
    int img_num = img_list.size();

    LOG(WARNING) << "EXECUTER !!!!!!!! ";

    Context<ARM> ctx(0, 0, 0);
    // do inference
    double to = 0;
    double tmin = 1000000;
    double tmax = 0;
    saber::SaberTimer<ARM> t1;
    // Tensor<ARM>* vtin = net_executer.get_in(vin_name[0]);
    Tensor<ARM>* vtin = net_executer.get_in_list()[0];
    // Tensor<ARM>* vtout = net_executer.get_out(out_name[0]);
    Tensor<ARM>* vtout = net_executer.get_out_list()[0];
    for (int i = 0; i < img_num; ++i){
        Mat im = imread(img_list[i]);
        CHECK_NOTNULL(im.data) << "read image " << img_list[i] << " failed";
        im = pre_process_img(im, vtin->width(), vtin->height());
        //resize(im, im, Size(vtin[0]->width(), vtin[0]->height()));
        im.convertTo(im, CV_32FC3, 1.f / 255);
        fill_tensor_with_cvmat(im, (float*)vtin->mutable_data(), 1, 3, vtin->width(), \
                               vtin->height(), mean_mb, scale_mb);
        //! net prediction
        Context<ARM> ctx2(0, 0, 0);
        t1.clear();
        t1.start(ctx2);
        net_executer.prediction();
        t1.end(ctx2);float tdiff = t1.get_average_ms();
        if (tdiff > tmax) {
            tmax = tdiff;
        }
        if (tdiff < tmin) {
            tmin = tdiff;
        }
        to += tdiff;
        int top1 = calc_top1((float*)vtout->mutable_data(), vtout->valid_size(), labels[i]);
        int top5 = calc_top5((float*)vtout->mutable_data(), vtout->valid_size(), labels[i]);
        top1_sum += top1;
        top5_sum += top5;
        LOG(INFO) <<"( "<< i << " ), " << img_list[i] << ",top1 accuracy: " << top1_sum / img_num \
            << ", top5 accuracy: " << top5_sum / img_num << ", prediction time: " << tdiff;
    }
    LOG(INFO) << "total, prediction time avg: " << to / img_num << ", min: " << tmin << ", max: " << tmax;
    //    std::string save_g_model_path = g_model_path + std::string(".saved");
    //    status = graph->save(save_g_model_path);
    delete graph;
}
#endif
/**
 * g_model_path 模型地址
 * g_batch_size batch大小,默认1
 * img_path 图像路径
 * label_path 标签路径
 * g_cluster 用到的核数,默认0， 大核
 * g_thread_num 用到的线程数,默认1
 * @param argc
 * @param argv
 * @return
 */

int main(int argc, const char** argv) {
    LOG(INFO)<< "usage:";
    LOG(INFO)<< argv[0] << " <anakin model> <num> <img_path> <img_file><cluster><threads>";
    LOG(INFO)<< "   lite_model:     path to anakin lite model";
    LOG(INFO)<< "   num:            batchSize default to 1";
    LOG(INFO)<< "   img_path:       images list path";
    LOG(INFO)<< "   img_file:       images list path";
    LOG(INFO)<< "   cluster:        choose which cluster to run, 0: big cores, 1: small cores, 2: all cores, 3: threads not bind to specify cores";
    LOG(INFO)<< "   threads:        set openmp threads";

    if (argc < 2) {
        LOG(ERROR) << "You should fill in the variable lite model at least.";
        return 0;
    }
    g_model_path = std::string(argv[1]);

    if (argc > 2) {
        g_batch_size = atoi(argv[2]);
    }
    if (argc > 3) {
        g_img_path = std::string(argv[3]);
    }
    if (argc > 4) {
        g_img_file= std::string(argv[4]);
    }
    if (argc > 5) {
        g_cluster = atoi(argv[5]);
        if (g_cluster < 0) {
            g_cluster = 0;
        }
        if (g_cluster > 5) {
            g_cluster = 5;
        }
    }
    if (argc > 6) {
        g_thread_num = atoi(argv[6]);
    }
    if (argc > 7) {
        g_set_archs = true;
        if (atoi(argv[7]) > 0) {
            g_arch = (ARMArch)atoi(argv[7]);
        } else {
            g_arch = ARM_UNKOWN;
        }
    }

    Env<ARM>::env_init();
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}
#else
int main(int argc, const char** argv) {
    return 0;
}
#endif
