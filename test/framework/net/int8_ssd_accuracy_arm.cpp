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
std::string g_test_list = "test_val.txt";
std::string g_img_file = "/data/local/tmp/ssd_data";
int g_thread_num = 1;
int g_cluster = 0;
float g_thresh = 0.5;
float g_val_iou = 0.5;
bool g_set_archs = false;
ARMArch g_arch = A73;

#ifdef USE_OPENCV
struct Object{
    int batch_id;
    cv::Rect rec;
    int class_id;
    float prob;
};

const char* class_names[] = {"background",
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};
//! set your mean value and scale value here
// ==================== precision ===================
    std::unordered_map<int, float> img_class_static;
    std::unordered_map<int, float> img_ap_static;
// ==================================================

void fill_tensor_with_cvmat(const Mat& img_in, Tensor<ARM>& tout, const int num, \
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
bool cal_iou(Object object, int xmin, int ymin, int xmax, int ymax, const float val_IOU){
    int sx = object.rec.x;
    int sy = object.rec.y;
    int ex = sx + object.rec.width;
    int ey = sy + object.rec.height;
    //交集
    int tar_sx = sx > xmin ? sx : xmin;
    int tar_sy = sy > ymin ? sy : ymin;
    int tar_ex = ex < xmax ? ex : xmax;
    int tar_ey = ey < ymax ? ey : ymax;
    // 并集
    int tar2_sx = sx < xmin ? sx : xmin;
    int tar2_sy = sy < ymin ? sy : ymin;
    int tar2_ex = ex > xmax ? ex : xmax;
    int tar2_ey = ey > ymax ? ey : ymax;

    int inter_size = (tar_ex - tar_sx) * (tar_ey - tar_sy);
    int union_size = (tar2_ex - tar2_sx) * (tar2_ey - tar2_sy);

    float ratio = (float) inter_size / (float) union_size;

    return ratio > val_IOU;

}
void detect_object(Tensor<ARM> tout, Mat& image, int num, const float thresh, \
    const float val_IOU, std::vector<std::string> labels) {
    std::vector<Object> objects;
    const float* dout = tout.data();
    for (int iw = 0; iw < tout.height(); iw++) {
        Object object;
        const float *values = dout + iw * tout.width();
        int batch_id = static_cast<int>(values[0]);
        int oriw = image.cols;
        int orih = image.rows;
        object.batch_id = batch_id;
        object.class_id = (int)values[1];
        object.prob = values[2];
        object.rec.x = (int)(values[3] * oriw);
        object.rec.y = (int)(values[4] * orih);
        object.rec.width = (int)(values[5] * oriw - object.rec.x);
        object.rec.height = (int)(values[6] * orih - object.rec.y);
        objects.push_back(object);
    }

    std::unordered_map<int, int> total_classes;
    std::unordered_map<int, int> part_classes;
    for (int i = 0; i< objects.size(); ++i) {
        Object object = objects.at(i);
        if (object.prob > thresh) {
            // cv::rectangle(image, object.rec, cv::Scalar(255, 0, 0));
            // std::ostringstream pro_str;
            // pro_str << object.prob;
            // std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            // cv::putText(image, label, cv::Point(object.rec.x, object.rec.y), \
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            LOG(INFO) << "detection in batch: " << object.batch_id << ", image size: " << image.cols << ", " << image.rows << \
                    ", detect object: " << class_names[object.class_id] << ", location: x=" << object.rec.x << ", y=" << object.rec.y << \
                      ", width=" << object.rec.width << ", height=" << object.rec.height;
            //read txt
            LOG(INFO) << "labels: " << labels[num];
            std::ifstream in(labels[num]);
            std::string line;
            while (getline(in, line)){
                std::istringstream ins(line);
                int id = 0;
                int xmin = 0;
                int ymin = 0;
                int xmax = 0;
                int ymax = 0;
                ins >> id >> xmin >> ymin >> xmax >> ymax;
                LOG(INFO) << "id: " << id << ", xmin: " << xmin << ", ymin: " << \
                            ymin << ", xmax: " << xmax << ", ymax: " << ymax;
                LOG(INFO) << "reinfrence id: " << object.class_id << ", x: " << object.rec.x << ", y: " << \
                            object.rec.y << ", height: " << object.rec.height << ", width: " << object.rec.width;
                typename std::unordered_map<int, int>::iterator it;
                //find class
                it = total_classes.find(id);
                if (it != total_classes.end()){
                    total_classes[id]++;
                }else{
                    total_classes[id] = 1;
                }
                bool tp = cal_iou(object, xmin, ymin, xmax, ymax, val_IOU);

                if (tp){
                    typename std::unordered_map<int, int>::iterator it2;
                    it2 = part_classes.find(id);
                    if (it2 != part_classes.end()){
                        part_classes[id]++;
                    }else{
                        part_classes[id] = 1;
                    }
                    printf("find sucess\n");
                }
                // printf("i: %d\n", i);
            }
            // cv::imwrite("detection_output.jpg", image);
        }
    }
    typename std::unordered_map<int, int>::iterator it;
    for (it = total_classes.begin(); it != total_classes.end(); it++){
        int key = it->first;
        if (img_class_static.find(key) != img_class_static.end()){
            img_class_static[key]++;
        }else{
            img_class_static[key] = 1.f;
        }
        if (part_classes.find(key) != part_classes.end()){
            float ratio = (float)part_classes[key] / (float)total_classes[key];
            LOG(INFO) <<", class id: " << key << ", class name: " << class_names[key] \
                    << ", precision: " << ratio;
            if (img_ap_static.find(key) != img_ap_static.end()){
                img_ap_static[key] += ratio;
            }else{
                img_ap_static[key] = ratio;
            }
        }
    }
}

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
    std::vector<std::string> img_list;
    std::vector<std::string> labels_list;
    //! load test image list
    std::fstream fp_img(g_test_list);
    std::string line;
    while (getline(fp_img, line)) {
        std::string path = line.substr(0, line.find("\t"));
        std::string label = line.substr(line.find("\t") + 1);
        path = g_img_file + path;
        LOG(INFO) << "img_file_path: " << path;
        img_list.push_back(path);
        label = g_img_file + label;
        LOG(INFO) << "label_path: " << label;
        labels_list.push_back(label);
    }
    int img_num = img_list.size();

    LOG(WARNING) << "EXECUTER !!!!!!!! ";

    Context<ARM> ctx(0, 0, 0);
    // do inference
    double to = 0;
    double tmin = 1000000;
    double tmax = 0;
    saber::SaberTimer<ARM> t1;
    Tensor<ARM>* vtin = net_executer.get_in_list()[0];
    Tensor<ARM>* vtout = net_executer.get_out_list()[0];
    float mean_mb[3] = {127.5f, 127.5f, 127.5f};
    float scale_mb[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    // img_num = 1;
    for (int i = 0; i < img_num; ++i){
        // Mat im = imread(img_list[i]);
        // CHECK_NOTNULL(im.data) << "read image " << img_list[i] << " failed";
        // // im = pre_process_img(im, vtin->width(), vtin->height());
        // //resize(im, im, Size(vtin[0]->width(), vtin[0]->height()));
        // im.convertTo(im, CV_32FC3, 1.f / 255);
        // fill_tensor_with_cvmat(im, (float*)vtin->mutable_data(), 1, 3, vtin->width(), \
                               vtin->height(), mean_mb, scale_mb);
        LOG(INFO) << "loading image " << img_list[i] << " ...";
        Mat img = imread(img_list[i], CV_LOAD_IMAGE_COLOR);
        if (img.empty()) {
            LOG(FATAL) << "opencv read image " << img_list[i] << " failed";
        }
        fill_tensor_with_cvmat(img, *vtin, g_batch_size, vtin->width(), vtin->height(), mean_mb, scale_mb);
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
        detect_object(*vtout, img, i, g_thresh, g_val_iou, labels_list);
        typename std::unordered_map<int, float>::iterator it;
        for (it = img_class_static.begin(); it != img_class_static.end(); it++){
            auto key = it->first;
            if (img_ap_static.find(key) != img_ap_static.end()){
                LOG(INFO) <<", class id: " << key << ", class name: " << class_names[key] << \
                            ", precision: " << img_ap_static[key] / img_class_static[key];
            }else{
                 LOG(INFO) <<", class id: " << key << ", class name: " << class_names[key] << \
                            "not in  ";
            }
        }
        LOG(INFO) <<"( "<< i << " ), " << img_list[i] << ", prediction time: " << tdiff;
    }
    typename std::unordered_map<int, float>::iterator it;
    float sum = 0.f;
    for (it = img_class_static.begin(); it != img_class_static.end(); it++){
        auto key = it->first;
        if (img_ap_static.find(key) != img_ap_static.end()){
            sum += img_ap_static[key] / img_class_static[key];
            LOG(INFO) <<", class id: " << key << ", class name: " << class_names[key] <<  \
                        ", precision: " << img_ap_static[key] << ", total number: " << img_class_static[key] \
                        << ", Averag_precision: " << img_ap_static[key] / img_class_static[key];
        }else{
                LOG(INFO) <<", class id: " << key << ", class name: " << class_names[key] << \
                            "not in  ";
        }
    }
    LOG(INFO) << "sum = " << sum << ", img_class_static size = " << img_class_static.size() \
            << ", this mAP = " << (sum / (float)img_class_static.size());
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
    LOG(INFO)<< argv[0] << " <anakin model> <num> <test_list> <img_file><cluster><threads><thresh><IOU>";
    LOG(INFO)<< "   lite_model:     path to anakin lite model";
    LOG(INFO)<< "   num:            batchSize default to 1";
    LOG(INFO)<< "   test_list:      test txt path";
    LOG(INFO)<< "   img_file:       images list path";
    LOG(INFO)<< "   cluster:        choose which cluster to run, 0: big cores, 1: small cores, 2: all cores, 3: threads not bind to specify cores";
    LOG(INFO)<< "   threads:        set openmp threads";
    LOG(INFO)<< "   thresh:        set detect_thresh";
    LOG(INFO)<< "   IOU:        set IOU thresh and cal mAP";

    if (argc < 2) {
        LOG(ERROR) << "You should fill in the variable lite model at least.";
        return 0;
    }
    g_model_path = std::string(argv[1]);

    if (argc > 2) {
        g_batch_size = atoi(argv[2]);
    }
    if (argc > 3) {
        g_test_list = std::string(argv[3]);
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
    if (argc > 7){
        g_thresh = (float)atof(argv[7]);
    }
    if (argc > 8){
        g_val_iou = (float)atof(argv[8]);
    }
    if (argc > 9) {
        g_set_archs = true;
        if (atoi(argv[9]) > 0) {
            g_arch = (ARMArch)atoi(argv[9]);
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
