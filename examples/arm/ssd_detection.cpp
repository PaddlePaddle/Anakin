#include "graph_base.h"
#include "graph.h"
#include "scheduler.h"
#include "net.h"
#include "worker.h"
#include "tensor_op.h"
#include "timer.h"

using namespace anakin::saber;
using namespace anakin::graph;
using namespace anakin;
typedef Tensor<ARM, AK_FLOAT, NCHW> Tensor4hf;

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

void detect_object(Tensor4hf& tout, const float thresh, Mat& image) {
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

    for (int i = 0; i< objects.size(); ++i) {
        Object object = objects.at(i);
        if (object.prob > thresh) {
            cv::rectangle(image, object.rec, cv::Scalar(255, 0, 0));
            std::ostringstream pro_str;
            pro_str << object.prob;
            std::string label = std::string(class_names[object.class_id]) + ": " + pro_str.str();
            cv::putText(image, label, cv::Point(object.rec.x, object.rec.y), \
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            LOG(INFO) << "detection in batch: " << object.batch_id << ", image size: " << image.cols << ", " << image.rows << \
                    ", detect object: " << class_names[object.class_id] << ", location: x=" << object.rec.x << ", y=" << object.rec.y << \
                      ", width=" << object.rec.width << ", height=" << object.rec.height;
            cv::imwrite("detection_output.jpg", image);
        }
    }
}
#endif

void test_net(const std::string model_file_name, const std::string image_file_name, float thresh, \
    int threads, int test_iter) {

    int batch_size = 1;

    //! create runtime context
    LOG(INFO) << "create runtime context";
    std::shared_ptr<Context<ARM>> ctx1 = std::make_shared<Context<ARM>>();
    ctx1->set_run_mode(SABER_POWER_HIGH, threads);
    LOG(INFO) << omp_get_num_threads() << " threads is activated";

    //! load model
    LOG(WARNING) << "load anakin model file from " << model_file_name << " ...";
    Graph<ARM, AK_FLOAT, Precision::FP32> graph;
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
    Net<ARM, AK_FLOAT, Precision::FP32, OpRunType::SYNC> net_executer(graph, ctx1, true);

    //! get in
    LOG(INFO) << "get input";
    auto d_tensor_in_p = net_executer.get_in("input_0");
    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i = 0; i < valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
    }
    Tensor4hf thin(valid_shape_in);

    //! feed input image to input tensor
#ifdef USE_OPENCV
    LOG(INFO) << "loading image " << image_file_name << " ...";
    Mat img = imread(image_file_name, CV_LOAD_IMAGE_COLOR);
    if (img.empty()) {
        LOG(FATAL) << "opencv read image " << image_file_name << " failed";
    }
    float mean_mb[3] = {127.5f, 127.5f, 127.5f};
    float scale_mb[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    fill_tensor_with_cvmat(img, thin, batch_size, thin.width(), thin.height(), mean_mb, scale_mb);
#else
    fill_tensor_host_const(thin, 1.f);
#endif

    //! do inference
    Context<ARM> ctx(0, 0, 0);
    anakin::saber::SaberTimer<ARM> my_time;
    LOG(INFO) << "run prediction ";

    double to = 0;
    double tmin = 1000000;
    double tmax = 0;
    my_time.start(ctx);
    saber::SaberTimer<ARM> t1;
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

    //! fixme get output
    //std::vector<Tensor4hf*> vout = net_executer.get_out_list();
    std::vector<Tensor4hf*> vout;
    for (auto& it : vout_name) {
        vout.push_back(net_executer.get_out(it));
    }
    Tensor4hf* tensor_out = vout[0];
    LOG(INFO) << "output size: " << vout.size();
#if 0 //print output data
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
#ifdef USE_OPENCV
    detect_object(*tensor_out, thresh, img);
#endif
}

int main(int argc, char** argv){

    LOG(INFO) << "initialized the device";
    Env<ARM>::env_init();

    if (argc < 2) {
        LOG(ERROR) << "usage: " << argv[0] << ": model_file image_name [detect_thresh] [test_iter] [threads]";
        return -1;
    }
    char* model_file = argv[1];

    char* image_path = argv[2];

    float thresh = 0.6;
    if(argc > 3) {
        thresh = (float)atof(argv[3]);
    }

    int test_iter = 10;
    if (argc > 4) {
        test_iter = atoi(argv[4]);
    }

    int threads = 1;
    if (argc > 5) {
        threads = atoi(argv[5]);
    }

	test_net(model_file, image_path, thresh, threads, test_iter);
    return 0;
}

