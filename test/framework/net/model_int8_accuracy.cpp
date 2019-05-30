#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include "saber/funcs/debug.h"
#include "saber/core/tensor_op.h"

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#define DEFINE_GLOBAL(type, var, value) \
		type (GLB_##var) = (value)

DEFINE_GLOBAL(int, gpu, 0);
DEFINE_GLOBAL(std::string, model_path, "");
DEFINE_GLOBAL(std::string, image_root, "");
DEFINE_GLOBAL(std::string, image_list, "");
DEFINE_GLOBAL(int, num, 1);
DEFINE_GLOBAL(int, img_num, -1);
DEFINE_GLOBAL(int, offset_y, 0);
DEFINE_GLOBAL(bool, graph_reset_bs, true);
DEFINE_GLOBAL(bool, rgb, false);
DEFINE_GLOBAL(bool, vis, false);

DEFINE_GLOBAL(std::string, input_data_source, "1");
DEFINE_GLOBAL(int, max_num, 32);
DEFINE_GLOBAL(bool, dynamic_batch, false);

#ifdef USE_OPENCV
template <typename TargetType>
void fill_tensor_with_cvmat(const cv::Mat& img_in, Tensor<TargetType>& tout, const int num, \
    const int width, const int height, const float* mean, const float* scale) {
    cv::Mat im;
    cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
    float* ptr_data_in = (float*)tout.mutable_data();
    int stride = width * height;
    for (int i = 0; i < num; i++) {
        float* ptr_in = ptr_data_in + i * tout.channel() * tout.height() * tout.width();
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                ptr_in[r * width + c] = (im.at<cv::Vec3b>(r, c)[2] - mean[0]) * scale[0];
                ptr_in[stride + r * width + c] = (im.at<cv::Vec3b>(r, c)[1] - mean[1]) * scale[1];
                ptr_in[2 * stride + r * width + c] = (im.at<cv::Vec3b>(r, c)[0] - mean[2]) * scale[2];
            }
        }
    }
}
#endif

void SplitString(const std::string& s,
        std::vector<std::string>& v, const std::string& c) {

    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2-pos1));
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length()) {
        v.push_back(s.substr(pos1));
    }
}

bool read_image_list(std::string &filename,
        std::vector<std::string> &results, std::vector<int> &label) {

    //std::cout << "image list: " << filename << std::endl;
    std::ifstream infile(filename.c_str());
    if (!infile.good()) {
        std::cout << "Cannot open " << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(infile, line)) {
        std::vector<std::string> v;
        SplitString(line, v, " ");
        if (v.size() < 2) {
            LOG(FATAL) << "wrong file list! [path label]";
        }
        results.push_back(v[0]);
        label.push_back(atoi(v[1].c_str()));
    }
    return true;
}

int print_topk(const float* scores, const int size, const int topk, \
    const std::vector<int>& labels) {

    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

//    LOG(INFO) << " out: " << vec[0].second <<" label: "<< labels[0];
    // print topk and score
    for (int i = 0; i < topk; i++) {
//        float score = vec[i].first;
//        int index = vec[i].second;
        if (vec[i].second == labels[0]) {
            return 1;
        }
//                LOG(INFO) << i <<": " << index << "  " << labels[index] << "  " << score;
    }
    return 0;
}

//! set your mean value and scale value here
//float mean_mb[3] = {103.939, 116.779, 123.68};
//float mean_mb[3] = {103.94, 116.78, 123.68};
//float scale_mb[3] = {1.f, 1.f, 1.f}; // for resnet
//float scale_mb[3] = {0.017, 0.017, 0.017}; // mobilenet

// fluid
float mean_mb[3] = {255.f * 0.485, 255.f * 0.456, 255.f * 0.406};
float scale_mb[3] = {1.f / 0.229 / 255.f, 1.f / 0.224f/255.f, 1.f / 0.225 / 255.f};

template <typename TargetType, typename TargetType_h>
void model_test() {
#ifdef USE_OPENCV
    using namespace cv;
#endif
    Graph<TargetType, Precision::FP32>* graph = new Graph<TargetType, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << GLB_model_path << " ...";

    // load anakin model files.
    auto status = graph->load(GLB_model_path);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    auto in_list = graph->get_ins();

    int max_batch_size = (GLB_max_num > GLB_num) ? GLB_max_num : GLB_num;
    int batch_size = GLB_num;

    //reshape shape batch-size
    // set batch
    const std::string& input_name = graph->get_ins().at(0);
    for (auto& in : graph->get_ins()) {
        graph->ResetBatchSize(in, max_batch_size);
    }
    LOG(INFO) << "set max_batch_size : " << max_batch_size;

    //anakin graph optimization
//    graph->load_layout_config("model_layout_config");
    graph->load_calibrator_config("net_pt_config", "calibrate_file.txt");
    graph->Optimize();

    // constructs the executer net
    Net<TargetType, Precision::FP32> net_executer(true);
    net_executer.init(*graph);
    // get in
    auto d_tensor_in_p = net_executer.get_in(input_name);
    d_tensor_in_p->set_num(batch_size);
    LOG(INFO) << "set batch_size : " << batch_size;
    if ( ! GLB_graph_reset_bs ) {
        // get in
        auto init_shape_in = d_tensor_in_p->valid_shape();
        Shape new_shape({GLB_num, init_shape_in[1], init_shape_in[2], init_shape_in[3]}, Layout_NCHW);
        d_tensor_in_p->reshape(new_shape);
    }

    Tensor4d<TargetType_h> h_tensor_in;
    Tensor<TargetType_h> out_host;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    int width = d_tensor_in_p->width();
    int height = d_tensor_in_p->height();
    int num = d_tensor_in_p->num();

    // ==================== precision ===================
    int top1_count = 0;
    int top5_count = 0;
    int total_count = 0;
    // ==================================================

//    for (int img_num = 0; img_num < image_file_list.size(); ++img_num)
    int new_batch_size = batch_size;
    std::vector<int> image_labels;
    char pro[102];
    memset(pro, '\0', sizeof(pro));
    const char* spin="-\\|/";
    int ratio = 0;
#ifdef USE_OPENCV
    std::vector<std::string> image_file_list;

    CHECK(read_image_list(GLB_image_list, image_file_list, image_labels));
    int image_file_list_size = image_file_list.size();
    total_count = image_file_list_size;
    if (GLB_img_num != -1) {
        image_file_list_size = GLB_img_num + 1;
    } else {
        GLB_img_num = 0;
    }

    for (int img_num = GLB_img_num; img_num < image_file_list_size; ++img_num)
#else
    int img_num = 0;
#endif
    {
        if (GLB_dynamic_batch) {
            new_batch_size = (img_num % (max_batch_size)) + 1;
        }
        d_tensor_in_p->set_num(new_batch_size);
        valid_shape_in = d_tensor_in_p->valid_shape();
        h_tensor_in.re_alloc(valid_shape_in);
        /*================fill tensor=================*/
#ifdef USE_OPENCV
        fflush(stdout);
        ratio = (int)(100.f * (float)img_num / (float)image_file_list_size);
        printf("[%-100s][%d\%][%c]\r", pro, ratio, spin[ratio & 3]);
        pro[ratio] = '=';

        std::string image_path = GLB_image_root + image_file_list[img_num];
//        LOG(INFO) << "loading image " << image_path << " ...";
        Mat img = imread(image_path, CV_LOAD_IMAGE_COLOR);
        if (img.empty()) {
            LOG(FATAL) << "opencv read image " << image_path << " failed";
        }

        // FOR NHWC
        if (h_tensor_in.width() == 3) {
            fill_tensor_with_cvmat(img, h_tensor_in, batch_size, h_tensor_in.height(),
                 h_tensor_in.channel(), mean_mb, scale_mb);
        } else {
            fill_tensor_with_cvmat(img, h_tensor_in, batch_size, h_tensor_in.width(),
                 h_tensor_in.height(), mean_mb, scale_mb);
        }
#else
        fill_tensor_const(h_tensor_in, 1.f);
#endif
        d_tensor_in_p->copy_from(h_tensor_in);
#ifdef USE_CUDA
        cudaDeviceSynchronize();
#endif
        std::string input_file_name = "record_In_0_image_";
        std::ostringstream ss;
        ss << input_file_name << img_num << ".txt";
        input_file_name = ss.str();
//        write_tensorfile(*d_tensor_in_p, input_file_name.c_str());
#ifdef USE_CUDA
        cudaDeviceSynchronize();
#endif
        /*================ launch =======================*/
        Context<TargetType> ctx(GLB_gpu, 0, 0);

        net_executer.prediction();
#ifdef USE_CUDA
        cudaDeviceSynchronize();
#endif
        /*=============no dump======================*/
        auto graph_outs = graph->get_outs();
        auto tensor_out_p = net_executer.get_out(graph_outs[0]);
        out_host.reshape(tensor_out_p->valid_shape());
        out_host.copy_from(*tensor_out_p);
#ifdef USE_CUDA
        cudaDeviceSynchronize();
#endif
        top1_count += print_topk((const float*)out_host.data(), 1000, 1, {image_labels[img_num]});
        top5_count += print_topk((const float*)out_host.data(), 1000, 5, {image_labels[img_num]});
//        for (int out_id = 0; out_id < graph_outs.size(); ++out_id) {
//            auto tensor_out_p = net_executer.get_out(graph_outs[out_id]);
//            write_tensorfile(*tensor_out_p,
//                    ("record_" + graph_outs[out_id] + "_image_" + std::to_string(img_num) + ".txt").c_str());
//        }
    }
    float top1 = (float)top1_count / (float)total_count;
    float top5 = (float)top5_count/ (float)total_count;
    LOG(INFO) << " top1: " << top1 << " top5: " << top5;
#ifndef ENABLE_DEBUG
    {
        auto d_tensor_in_p = net_executer.get_in(input_name);
        //Shape new_shape({1, 14, 800, 1408});
        //d_tensor_in_p->reshape(new_shape);
        // performance check
        int warm_up = 100;
        int ts = 1000;
        for (int i = 0; i < warm_up; ++i) {
            net_executer.prediction();
        }
#ifdef USE_CUDA
        cudaDeviceSynchronize();
#endif
        Context<TargetType> ctx(GLB_gpu, 0, 0);
        saber::SaberTimer<TargetType> my_time;
        for (int i = 0; i < ts; ++i) {
            my_time.start(ctx);
            net_executer.prediction();
#ifdef USE_CUDA
            cudaDeviceSynchronize();
#endif
            my_time.end(ctx);
        }
        std::cout << "==========================Performance Statistics =============================\n";
        std::cout << "==================== Input_shape:       ["
                  << d_tensor_in_p->num() << ", "
                  << d_tensor_in_p->channel() << ", "
                  << d_tensor_in_p->height() << ", "
                  << d_tensor_in_p->width() << "]\n";
        std::cout << "==================== Warm_up:           " << warm_up << "\n";
        std::cout << "==================== Iteration:         " << ts << "\n";
        std::cout << "==================== Average time:      " << my_time.get_average_ms()  << "ms\n";
        std::cout << "==================== 10% Quantile time: " << my_time.get_tile_time(10) << "ms\n";
        std::cout << "==================== 25% Quantile time: " << my_time.get_tile_time(25) << "ms\n";
        std::cout << "==================== 50% Quantile time: " << my_time.get_tile_time(50) << "ms\n";
        std::cout << "==================== 75% Quantile time: " << my_time.get_tile_time(75) << "ms\n";
        std::cout << "==================== 90% Quantile time: " << my_time.get_tile_time(90) << "ms\n";
        std::cout << "==================== 95% Quantile time: " << my_time.get_tile_time(95) << "ms\n";
        std::cout << "==================== 99% Quantile time: " << my_time.get_tile_time(99) << "ms" << std::endl;
    }
#endif
    delete graph;
}

TEST(NetTest, net_execute_base_test) {
#ifdef USE_CUDA
    model_test<NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
    model_test<X86, X86>();
#endif
}

int main(int argc, const char** argv) {
#ifdef USE_OPENCV
    if (argc < 4) {
        LOG(FATAL) << "bad param \n ./anakin_model_test + model_path + img_root + img_list + [batch]";
    } else if (argc >= 4) {
        GLB_model_path = argv[1];
        GLB_image_root = argv[2];
        GLB_image_list = argv[3];
    }
    GLB_num = argc >= 5 ? atoi(argv[4]) : 1;
    GLB_gpu = argc >= 6 ? atoi(argv[5]) : 0;
    GLB_img_num = argc >= 7 ? atoi(argv[6]) : -1;
#else
    if (argc < 2) {
        LOG(FATAL) << "bad param \n ./anakin_model_test + model_path + [batch]";
    } else if (argc >= 2) {
        GLB_model_path = argv[1];
    }
#endif

    LOG(INFO) << " model path: " << GLB_model_path;
    LOG(INFO) << " image root: " << GLB_image_root;
    LOG(INFO) << " image list: " << GLB_image_list;
    LOG(INFO) << " GLB_num: " << GLB_num;
    LOG(INFO) << " using GPU: " << GLB_gpu;

#ifdef USE_CUDA
    cudaSetDevice(GLB_gpu);
    anakin::saber::Env<NV>::env_init();
    anakin::saber::Env<NVHX86>::env_init();
    cudaSetDevice(GLB_gpu);
#endif

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
