#include "test_lite.h"
#include "saber/lite/net/net_lite.h"
#include "saber/lite/net/saber_factory_lite.h"

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace anakin::saber;
using namespace anakin::saber::lite;
typedef Tensor<CPU> TensorHf;

std::string g_lite_model;
std::string g_img_list;
std::string g_gt_list;
int FLAGS_threads = 1;
int FLAGS_cluster = 0;
bool FLAGS_set_archs = false;
ARMArch FLAGS_arch = A73;

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

TEST(TestSaberLite, test_seg_precision) {

    std::vector<std::string> img_list;
    std::vector<std::string> gt_list;
    //! load test image list and ground truth image list
    std::fstream fp_img(g_img_list);
    std::string line;
    while (getline(fp_img, line)) {
        img_list.push_back(line);
    }
    LOG(INFO) << "total test image number: " << img_list.size();
    fp_img.close();

    std::fstream fp_gt(g_gt_list);
    while (getline(fp_gt, line)) {
        gt_list.push_back(line);
    }
    LOG(INFO) << "total ground truth image number: " << gt_list.size();
    CHECK_EQ(gt_list.size(), img_list.size()) << "test image number must = ground truth image number";

    LOG(INFO) << "finish load test image list";

    //! create net, with power mode and threads
    Net net((PowerMode)FLAGS_cluster, FLAGS_threads);
    //! you can also set net param according to your device
    net.set_run_mode((PowerMode)FLAGS_cluster, FLAGS_threads);
    if (FLAGS_set_archs) {
        net.set_device_arch(FLAGS_arch);
        LOG(INFO) << "arm arc: " << FLAGS_arch;
    }
    net.set_device_cache(32 * 1024, 512* 1024);
    //! load merged model
    SaberStatus flag = net.load_model(g_lite_model.c_str());
    CHECK_EQ(flag, SaberSuccess) << "load model: " << g_lite_model << " failed";
    LOG(INFO) << "load model: " << g_lite_model << " successed";

    std::vector<TensorHf*> vtin = net.get_input();
    LOG(INFO) << "number of input tensor: " << vtin.size();
    for (int i = 0; i < vtin.size(); ++i) {
        TensorHf* tin = vtin[i];
        //! reshape input before prediction
        Shape shin = tin->valid_shape();
        LOG(INFO) << "input tensor size: ";
        for (int j = 0; j < tin->dims(); ++j) {
            LOG(INFO) << "|---: " << shin[j];
        }
    }

    int hin = vtin[0]->height();
    int win = vtin[0]->width();

    //! change here according to your own model
    std::vector<TensorHf*> vtout = net.get_output();
    LOG(INFO) << "number of output tensor: " << vtout.size();
    for (int i = 0; i < vtout.size(); i++) {
        TensorHf* tout = vtout[i];
        LOG(INFO) << "output tensor size: ";
        Shape shout = tout->valid_shape();
        for (int j = 0; j < tout->dims(); ++j) {
            LOG(INFO) << "|---: " << shout[j];
        }
    }

    float mean_val[3] = {104.008f, 116.669f, 122.675f};
    float scale_val[3] = {1.f, 1.f, 1.f};

    double acc = 0.0;
    double max_acc = 0.0;
    double min_acc = 1.0;

    for (int k = 0; k < img_list.size(); ++k) {
        //! pre-processing
        Mat img = imread(img_list[k], CV_LOAD_IMAGE_COLOR);
        fill_tensor_with_cvmat(img, *vtin[0], 1, win, hin, mean_val, scale_val);
        LOG(INFO) << "test image name: " << img_list[k] << ", gt image name: " << gt_list[k];
        Mat img_gt = imread(gt_list[k], CV_LOAD_IMAGE_UNCHANGED);
        if (img.empty() || img_gt.empty()) {
            LOG(FATAL) << "load image failed";
        }
        Mat img_gt_resize;
        cv::resize(img_gt, img_gt_resize, cv::Size(192, 192));
        double to = 0;
        SaberTimer t1;
        t1.start();
        net.prediction();
        t1.end();
        to = t1.get_average_ms();
        LOG(INFO) << "time consumption: " << to << " ms";
//        for (int i = 0; i < vtout.size(); ++i) {
//            double mean = tensor_mean(*vtout[i]);
//            LOG(INFO) << "output mean: " << mean;
//        }

        //! post processing
        long long diff_count = 0;
        double acc_curr = 0.0;
        cmp_seg_result(img_gt_resize, *vtout[0], diff_count, acc_curr);
        acc += acc_curr;
        max_acc = max_acc > acc_curr ? max_acc : acc_curr;
        min_acc = min_acc < acc_curr ? min_acc : acc_curr;
        LOG(INFO) << "image : " << img_list[k] << ", diff count: " << diff_count << ", accuracy: " << acc_curr;
    }
    LOG(INFO) << "test accuracy is: " << acc / img_list.size() << ", min: " << min_acc << ", max: " << max_acc;
}

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);

    Env::env_init();

    LOG(INFO)<< "usage:";
    LOG(INFO)<< argv[0] << " <lite model> <image_list> <ground_truth_image_list> <threads[default 1]>";
    LOG(INFO)<< "   lite_model:     path to anakin lite model";
    LOG(INFO)<< "   image_list:     path to test image list";
    LOG(INFO)<< "   gt_image_list:  path to test image ground truth list";
    LOG(INFO)<< "   threads:        set openmp threads";
    if(argc < 4) {
        LOG(ERROR)<< argv[0] << " <lite model> <image_list> <ground_truth_image_list> <threads[default 1]>";
        return 0;
    }
    g_lite_model = argv[1];
    g_img_list = argv[2];
    g_gt_list = argv[3];

    if (argc > 4) {
        FLAGS_threads = atoi(argv[4]);
    }
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
#else
int main(int argc, const char** argv) {
    LOG(ERROR)<< "turn on opencv";
    return 0;
}
#endif //USE_OPENCV
