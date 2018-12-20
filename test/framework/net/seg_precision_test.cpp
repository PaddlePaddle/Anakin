#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include <fstream>

#if defined(USE_CUDA)
using Target = NV;
using Target_H = X86;
#elif defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#elif defined(USE_ARM_PLACE)
using Target = ARM;
using Target_H = ARM;
#elif defined(AMD_GPU)
using Target = AMD;
using Target_H = X86;
#endif

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
using namespace cv;
std::string g_model_path = "path/to/your/anakin_model";
std::string g_precition_path = "path/to/your/precision_file";
std::string g_calibrate_path = "path/to/your/calib_file";
std::string g_img_path = "path/to/your/image list";
std::string g_gt_path = "path/to/your/ground truth list";

typedef Tensor<X86> TensorHf;

void fill_tensor_with_cvmat(const Mat& img_in, TensorHf& tout, const int num, \
    const int width, const int height, const float* mean, const float* scale) {
    cv::Mat im;
    cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
    float* ptr_data_in = static_cast<float*>(tout.mutable_data());
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

void cmp_seg_result(const Mat& gt_img, const TensorHf& tin, long long& diff_count, double& accuracy) {
    int height = tin.height();
    int width = tin.width();
    diff_count = 0;
    const float* din = static_cast<const float*>(tin.data());
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int gt = gt_img.at<char>(h, w);
            int test = *(din++) >= 0.5;
            if (gt != test) {
                diff_count++;
            }
        }
    }
    accuracy = (double)diff_count / (height * width);
}

#ifdef USE_CUDA
TEST(NetTest, net_execute_base_test) {

    std::vector<std::string> img_list;
    std::vector<std::string> gt_list;
    //! load test image list and ground truth image list
    std::fstream fp_img(g_img_path);
    std::string line;
    while (getline(fp_img, line)) {
        img_list.push_back(line);
    }
    LOG(INFO) << "total test image number: " << img_list.size();
    fp_img.close();

    std::fstream fp_gt(g_gt_path);
    while (getline(fp_gt, line)) {
        gt_list.push_back(line);
    }
    LOG(INFO) << "total ground truth image number: " << gt_list.size();
    CHECK_EQ(gt_list.size(), img_list.size()) << "test image number must = ground truth image number";

    LOG(INFO) << "finish load test image list";

    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if (!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    //anakin graph optimization
    graph->Optimize();
    Net<NV, Precision::FP32> net_executer(true);
    net_executer.load_calibrator_config(g_precition_path, g_calibrate_path);
    net_executer.init(*graph);
    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    auto d_tensor_out_p = net_executer.get_out_list()[0];
    auto valid_shape_out = d_tensor_out_p->valid_shape();

    TensorHf h_tensor_in;
    h_tensor_in.re_alloc(valid_shape_in);

    TensorHf h_tensor_out;
    h_tensor_out.re_alloc(valid_shape_out);

    int hin = h_tensor_in.height();
    int win = h_tensor_in.width();

    float mean_val[3] = {104.008f, 116.669f, 122.675f};
    float scale_val[3] = {1.f, 1.f, 1.f};

    double acc = 0.0;

    for (int k = 0; k < img_list.size(); ++k) {
        //! pre-processing
        Mat img = imread(img_list[k], CV_LOAD_IMAGE_COLOR);
        fill_tensor_with_cvmat(img, h_tensor_in, 1, win, hin, mean_val, scale_val);
        LOG(INFO) << "test image name: " << img_list[k] << ", gt image name: " << gt_list[k];
        Mat img_gt = imread(gt_list[k], CV_LOAD_IMAGE_UNCHANGED);
        if (img.empty() || img_gt.empty()) {
            LOG(FATAL) << "load image failed";
        }
        Mat img_gt_resize;
        cv::resize(img_gt, img_gt_resize, cv::Size(192, 192));
        d_tensor_in_p->copy_from(h_tensor_in);

        net_executer.prediction();

        TargetWrapper<Target>::device_sync();
        h_tensor_out.copy_from(*d_tensor_out_p);

        double mean = tensor_mean_value_valid(h_tensor_out);
        LOG(INFO) << "output mean: " << mean;

        //! post processing
        long long diff_count = 0;
        double acc_curr = 0.0;
        cmp_seg_result(img_gt_resize, h_tensor_out, diff_count, acc_curr);
        acc += acc_curr;
        LOG(INFO) << "image : " << img_list[k] << ", diff count: " << diff_count << ", accuracy: " << acc_curr;
    }
    LOG(INFO) << "test accuracy is: " << acc / img_list.size();
}
#endif 

int main(int argc, const char** argv){
    if (argc < 6){
        LOG(ERROR) << "usage: " << argv[0] << " <model path> <precition file> <calib file> <image list> <ground truth list>";
        return 0;
    }
    g_model_path = std::string(argv[1]);
    g_precition_path = std::string(argv[2]);
    g_calibrate_path = std::string(argv[3]);
    g_img_path = std::string(argv[4]);
    g_gt_path = std::string(argv[5]);

    Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
#else //opencv
int main(int argc, const char** argv){
    LOG(ERROR) << "turn on USE_OPENCV firstly";
    return 0;
}
#endif //opencv