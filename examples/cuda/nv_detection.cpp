#include <string>
#include "framework/core/net/net.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "debug.h"
#include <fstream>
using namespace anakin::saber;
using namespace anakin::graph;
using namespace anakin;
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
typedef Tensor<Target_H> Tensor4hf;

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"

using namespace cv;

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

void fill_tensor_with_cvmat(const std::vector<Mat>& img_in, Tensor4hf& tout, const int num, \
    const int width, const int height, const float* mean, const float* scale) {
    CHECK_GE(img_in.size(), 1) << "must have at least one image";
    cv::Mat im;
    auto shape = tout.valid_shape();
    shape.set_height(height);
    shape.set_width(width);
    tout.reshape(shape);
    float* ptr_data_in = tout.mutable_data();
    int cstride = width * height;
    int nstride = tout.channel() * cstride;

    for (int i = 0; i < num; i++) {
        float* ptr_in = ptr_data_in + i * nstride;
        if (i < img_in.size()) {
            cv::resize(img_in[i], im, cv::Size(width, height), 0.f, 0.f);
            for (int r = 0; r < height; r++) {
                float* ptr_in_c0 = ptr_in + r * width;
                float* ptr_in_c1 = ptr_in_c0 + cstride;
                float* ptr_in_c2 = ptr_in_c1 + cstride;
                for (int c = 0; c < width; c++) {
                    ptr_in_c0[c] = (im.at<cv::Vec3b>(r, c)[0] - mean[0]) * scale[0];
                    ptr_in_c1[c] = (im.at<cv::Vec3b>(r, c)[1] - mean[1]) * scale[1];
                    ptr_in_c2[c] = (im.at<cv::Vec3b>(r, c)[2] - mean[2]) * scale[2];
                }
            }
        } else {
            memcpy(ptr_in, ptr_in - nstride, nstride * sizeof(float));
        }
    }
}

void detect_object(Tensor4hf& tout, const float thresh, std::vector<Mat>& image) {
    int img_num = image.size();
    const float* dout = static_cast<const float*>(tout.data());
    std::vector<Object> objects;
    for (int iw = 0; iw < tout.height(); iw++) {
        Object object;
        const float *values = dout + iw * tout.width();
        int batch_id = static_cast<int>(values[0]);
        int oriw = image[batch_id].cols;
        int orih = image[batch_id].rows;
        object.batch_id = batch_id;
        object.class_id = (int)values[1];
        object.prob = values[2];
        object.rec.x = (int)(values[3] * oriw);
        object.rec.y = (int)(values[4] * orih);
        object.rec.width = (int)(values[5] * oriw - object.rec.x);
        object.rec.height = (int)(values[6] * orih - object.rec.y);
        objects.push_back(object);
    }

    for (int i = 0; i < objects.size(); ++i) {
        Object object = objects.at(i);
        if (object.prob > thresh && object.batch_id < image.size()) {
            cv::rectangle(image[object.batch_id], object.rec, cv::Scalar(255, 0, 0));
            std::ostringstream pro_str;
            pro_str << object.prob;
            std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            cv::putText(image[object.batch_id], label, cv::Point(object.rec.x, object.rec.y), \
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            LOG(INFO) << "detection in batch: " << object.batch_id << ", image size: " << \
                    image[object.batch_id].cols << ", " << image[object.batch_id].rows << \
                    ", detect object: " << class_names[object.class_id] << ", location: x=" << \
                    object.rec.x << ", y=" << object.rec.y << ", width=" << object.rec.width << \
                    ", height=" << object.rec.height;
        }
    }
    for (int j = 0; j < image.size(); ++j) {
        std::ostringstream str;
        str << "detection_out_" << j << ".jpg";
        cv::imwrite(str.str(), image[j]);
    }
}
#endif

void test_net(const std::string model_file_name, const std::string image_file_name, float thresh, \
    int batch_size, int device_id) {

    Env<Target>::env_init();
    Env<Target_H>::env_init();
    TargetWrapper<Target>::set_device(device_id);

    //! load model
    LOG(INFO) << "load anakin model file from " << model_file_name << " ...";
    Graph<Target, Precision::FP32> graph;
    auto status = graph.load(model_file_name);
    if (!status) {
         LOG(FATAL) << " [ERROR] " << status.info();
    }

    auto ins_name = graph.get_ins();
    //! set batch size
    for (auto& in : ins_name) {
        graph.ResetBatchSize(in, batch_size);
    }

    //! optimize the graph
    LOG(INFO) << "optimize the graph";
    graph.Optimize();

    //! get output name
    std::vector<std::string>& vout_name = graph.get_outs();
    LOG(INFO) << "output size: " << vout_name.size();

    //! constructs the executer net
    LOG(INFO) << "create net to execute";
    Net<Target, Precision::FP32, OpRunType::SYNC> net_executer(graph, true);

#ifdef USE_OPENCV
    std::vector<Mat> img_list;
#endif

    //! get in
    auto d_tensor_in_p = net_executer.get_in_list();
    auto d_tensor_out_p = net_executer.get_out_list();
    for (auto& din : d_tensor_in_p) {
        auto valid_shape_in = din->valid_shape();
        for (int i = 0; i < valid_shape_in.size(); i++) {
            LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
        }
        Tensor4hf thin(valid_shape_in);
        //! feed input image to input tensor
#ifdef USE_OPENCV
        std::fstream fp(image_file_name);
        std::string line;
        std::vector<std::string> img_file_list;
        while (getline(fp, line)) {
            img_file_list.push_back(line);
        }
        LOG(INFO) << "total test image number: " << img_file_list.size();
        for (int i = 0; i < img_file_list.size(); ++i) {
            LOG(INFO) << "loading image : " << img_file_list[i];
            Mat img = imread(img_file_list[i], CV_LOAD_IMAGE_COLOR);
            if (img.empty()) {
                LOG(FATAL) << "opencv read image " << image_file_name << " failed";
            }
            img_list.push_back(img);
        }
        float mean_mb[3] = {104.f, 117.f, 123.f};
        float scale_mb[3] = {1.f, 1.f, 1.f};
        fill_tensor_with_cvmat(img_list, thin, batch_size, thin.width(), thin.height(), mean_mb, scale_mb);
        din->copy_from(thin);
#else
        fill_tensor_const(*din, 1.f);
#endif
    }


    //! do inference
    LOG(INFO) << "run prediction ";
    net_executer.prediction();


    LOG(INFO) << "finish infer: " << model_file_name << ", batch_size " << batch_size;

    //! fixme get output
    std::vector<Tensor4hf> vout;
    for (int i = 0; i < d_tensor_out_p.size(); i++) {
        Tensor4hf hout(d_tensor_out_p[i]->valid_shape());
        hout.copy_from(*d_tensor_out_p[i]);
        vout.push_back(hout);
    }
    Tensor4hf tensor_out = vout[0];
    LOG(INFO) << "output size: " << vout.size();
#if 1 //print output data
    LOG(INFO) << "extract data: size: " << tensor_out.valid_size() << \
        ", width=" << tensor_out.width() << ", height=" << tensor_out.height();
    const float* ptr_out = static_cast<const float*>(tensor_out.data());
    for (int i = 0; i < tensor_out.valid_size(); i++) {
        printf("%0.4f  ", ptr_out[i]);
        if ((i + 1) % 7 == 0) {
            printf("\n");
        }
    }
    printf("\n");
#endif
#ifdef USE_OPENCV
    detect_object(tensor_out, thresh, img_list);
#endif
}

int main(int argc, char** argv){

    logger::init(argv[0]);
    if (argc < 2) {
        LOG(ERROR) << "usage: " << argv[0] << ": model_file image_name [detect_thresh] [batch size] [device id]";
        return -1;
    }
    char* model_file = argv[1];

    char* image_path = argv[2];

    float thresh = 0.6;
    if(argc > 3) {
        thresh = (float)atof(argv[3]);
    }

    int batch_size = 1;
    if (argc > 4) {
        batch_size = atoi(argv[4]);
    }

    int device_id = 0;
    if (argc > 5) {
        device_id = atoi(argv[5]);
    }

	test_net(model_file, image_path, thresh, batch_size, device_id);
    return 0;
}

