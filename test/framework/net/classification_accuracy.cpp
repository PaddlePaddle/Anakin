#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include "net_test.h"
#include "framework/utils/data_common.h"
#include "saber/funcs/timer.h"
#include <chrono>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#ifdef USE_CUDA
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

//#define USE_DIEPSE
/*The file list is stored in data_file.
 * 1.txt
 * 2.txt*/
/*img and label is stored in 1.txt
 * 0000.jpeg 0
 * 0001.jpeg 1
 * 0002.jpeg 1
 * 0003.jpeg 2
 * */
std::string FLAGS_model_path;
std::string FLAGS_img_root;
std::string FLAGS_data_file;
int FLAGS_num = 1;
int FLAGS_warmup_iter = 10;
int FLAGS_epoch = 1000;
int FLAGS_left = 0;
int FLAGS_right = 0;
int FLAGS_top = 0;
int FLAGS_botoom = 0;
bool FLAGS_is_NCHW = false;
bool FLAGS_is_rgb = true;
float FLAGS_input_scale = 1.0f;
int FLAGS_height = 1;
int FLAGS_width = 1;


class Point {
public:
   Point(int x_in, int y_in) :x(x_in), y(y_in){}
   Point(const Point& right) : x(right.x), y(right.y) {}
   Point& operator = (const Point& right) {
        x = right.x;
        y = right.y;
        return *this;
   }
   ~Point() {}

   int x;
   int y;
};

class Rect {
public:
    Rect(Point lt_point_in, Point rt_point_in):lt_point(lt_point_in), 
         rb_point(rt_point_in) {}
    Rect(const Rect& right): lt_point(right.lt_point), rb_point(right.rb_point) {}
    Rect& operator = (const Rect& right) {
         lt_point = right.lt_point;
         rb_point = right.rb_point;
         return *this;
    }
    ~Rect() {}

    Point lt_point;
    Point rb_point;
};
bool my_func(std::pair<int, float> x, std::pair<int, float> y) {return (x.second > y.second); }

//#ifdef USE_OPENCV 
#ifdef IMG

void fill_image_data(const cv::Mat& img, float* cpu_data, bool is_NCHW, float scale) {
    int idx = 0;
    if (is_NCHW) {
        for(int c = 0; c < img.channels(); c++){
            for(int h = 0; h < img.rows; h++){
                for(int w = 0; w < img.cols; w++)
                    cpu_data[idx++] = img.data[h * img.step + w * img.channels() + c] * scale;
            }
        }
    } else {
        for (int h = 0; h < img.rows; h++) {
            for (int w = 0; w < img.cols; w++) {
                for (int c = 0; c < img.channels(); c++) {
                    cpu_data[idx++] = img.data[h * img.step + w * img.channels() + c] * scale;
                }
            }
        }
    }
}

template<typename Ttype, Precision Ptype>
void test_accuracy(std::string model_path,
                   std::string img_root,
                   std::string data_file,
                   Rect rect,
                   bool is_NCHW,
                   bool is_rgb,
                   float input_scale,
                   int height,
                   int width) {
    Graph<Ttype, Ptype>* graph = new Graph<Ttype, Ptype>();
    auto status = graph->load(model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    Net<Ttype, Ptype> net_executer(*graph, true);

    auto d_tensor_in_p = net_executer.get_in("input_0");
    auto d_tensor_out_p = net_executer.get_out("pred");

    Tensor4d<Target_H> h_tensor_in;
    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i = 0; i < valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    Tensor4d<Target_H> h_tensor_out;
    auto valid_shape_out = d_tensor_out_p->valid_shape();
    for (int i = 0; i < valid_shape_out.size(); i++) {
        LOG(INFO) << "detect out dims[" << i << "]" << valid_shape_out[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    h_tensor_out.re_alloc(valid_shape_out);

    
    std::ifstream ifs;
    ifs.open(data_file);
    int img_num = 0;
    int top1_num = 0; 
    int top5_num = 0;
    float* cpu_data = (float*) h_tensor_in.mutable_data();
    while(true) {
        std::string file_name;
        std::getline (ifs, file_name);
        if (file_name != "") {
            break;
        }
        std::ifstream ifs_img(file_name);
        std::string img_label;
        while (true) {
            std::getline(ifs_img, img_label);
            if (img_label != " ") {
                break;
            }
            std::vector<std::string> vec = string_split(img_label, " ");
            std::string img_name = vec[0];
            int img_label_index = atoi(vec[1].c_str());
            std::string img_path = img_root + img_name;
            cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
            cv::Rect roi(rect.lt_point.x, rect.lt_point.y, 
                rect.rb_point.x -rect.lt_point.x, rect.rb_point.y - rect.lt_point.y);
            cv::Mat img_roi = img(roi);
            if (img_roi.data == 0) {
                LOG(ERROR) << "Failed to read iamge: " << img_path;
                return -1;
            }
            cv::resize(img_roi, img_roi, cv::Size(width, height));
            
            if (is_rgb) {
                cv::cvtColor(img_roi, img_roi, CV_BGR2RGB);
            }
            fill_image_data(img_roi, cpu_data, is_NCHW, input_scale);
            d_tensor_in_p->copy_from(h_tensor_in);
            net_executer.prediction();
            
            #ifdef USE_CUDA
                cudaDeviceSynchronize();
            #endif
            h_tensor_out.copy_from(*d_tensor_out_p);
            const float* data = (const float*) h_tensor_out.data();
            std::vector<std::pair<int, float >> index_prob_vec;
            for (int i = 0 ; i < h_tensor_out.valid_size(); i++) {
                index_prob_vec.push_back(std::make_pair(i, float(data[i])));
            }
            std::partial_sort(index_prob_vec.begin(), index_prob_vec.begin() + 5, index_prob_vec.end(), my_func);
            if (index_prob_vec[0].first == img_label_index) {
                top1_num++;
            }
            for (int i = 0; i < 5; i++) {
                if (index_prob_vec[i].first == img_label_index) {
                    top5_num++;
                    break;
                }
            }
            img_num++;
        }
    }
    //LOG(INFO) << model_path <<" " << Ptype << " top_1 accuracy: " << top1_num * 1.0f / img_num;
    //LOG(INFO) << model_path <<" " << Ptype << " top_5 accuracy: " << top5_num * 1.0f / img_num;
    LOG(INFO) << model_path <<" "  << " top_1 accuracy: " << top1_num * 1.0f / img_num;
    LOG(INFO) << model_path <<" "  << " top_5 accuracy: " << top5_num * 1.0f / img_num;
}
#endif
/*data format 
 * label txt
 * 3
 * 5 
 * 8
 * data txt
 * 128
 * 109
 * 255
 */
 

template<typename Ttype, Precision Ptype>
void test_accuracy(std::string model_path,
                   std::string data_file,
                   std::string label_file) {
     std::ifstream ifs_data;
     std::ifstream ifs_label;
     float mean[3] = {125.307, 122.95, 113.865};
     float scale = 1.f;
     ifs_data.open(data_file);
     ifs_label.open(label_file);
     CHECK(ifs_data.is_open()) << "data file can not be opened";
     CHECK(ifs_label.is_open()) << "label file can not be opened";
     int input_size = 3072;
     int batch_size = 1;
     std::vector<int> labels;
     std::string str;
     while(true) {
         getline(ifs_label, str);
         if (str == "") {
            break;
         }
         labels.push_back(atoi(str.c_str()));
     }
    Graph<Ttype, Ptype>* graph = new Graph<Ttype, Ptype>();
    auto status = graph->load(model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    graph->Optimize();

    Net<Ttype, Ptype> net_executer(true);
    net_executer.init(*graph);
    

    auto d_tensor_in_p = net_executer.get_in("input_0");
    auto d_tensor_out_p = net_executer.get_out("ip1_out");
    Tensor4d<Target_H> h_tensor_in;
    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i = 0; i < valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    Tensor4d<Target_H> h_tensor_out;
    auto valid_shape_out = d_tensor_out_p->valid_shape();
    for (int i = 0; i < valid_shape_out.size(); i++) {
        LOG(INFO) << "detect out dims[" << i << "]" << valid_shape_out[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    h_tensor_out.re_alloc(valid_shape_out);
    auto h_data = (float*) h_tensor_in.mutable_data();
    int top1_num = 0;
    int top5_num = 0;
    int img_num = 0;
    int img_size = 32*32;
    while(true) {
         int i = 0;
         while(i < h_tensor_in.valid_size()) {
             getline(ifs_data, str);
             if (str == "") {
                break;
             }
             auto mean_id = i / img_size;
             h_data[i] = scale * (atof(str.c_str()) - mean[mean_id]);
             //h_data[i] = atof(str.c_str());
             i++;
         }
         if (i == 0) {
            break;
         }
         d_tensor_in_p->copy_from(h_tensor_in);
         net_executer.prediction();
         h_tensor_out.copy_from(*d_tensor_out_p);
         const float* data = (const float*) h_tensor_out.data();
         std::vector<std::pair<int, float >> index_prob_vec;
         for (int i = 0 ; i < h_tensor_out.valid_size(); i++) {
             index_prob_vec.push_back(std::make_pair(i, float(data[i])));
         }
         std::partial_sort(index_prob_vec.begin(), index_prob_vec.begin() + 5, index_prob_vec.end(), my_func);
        
         if (index_prob_vec[0].first == labels[img_num]) {
             top1_num++;
         }
         for (int i = 0; i < 5; i++) {
             if (index_prob_vec[i].first == labels[img_num]) {
                 top5_num++;
                 break;
             }
         }
         img_num++;
    }
    if (img_num != 0) {
        LOG(INFO) << " top1 " << top1_num * 1.0f /img_num << " top5 " << top5_num * 1.0f/img_num;
    } else {
        LOG(INFO)<< "img_num is zero";
    }
    delete graph;
}

TEST(NetTest, net_execute_base_test) {

#if defined(USE_CUDA) && defined(USE_OPENCV)
//test_accuracy<NV,Precision::FP32>( FLAGS_model_path,
//               FLAGS_img_root,
//               FLAGS_data_file,
//               Rect(Point(FLAGS_left, FLAGS_top), Point(FLAGS_right, FLAGS_right)),
//               FLAGS_is_NCHW,
//               FLAGS_is_rgb,
//               FLAGS_input_scale,
//               FLAGS_height,
//               FLAGS_width);
#endif
#if defined(NVIDIA_GPU)
   std::string data_file = "/home/zhangshuai20/workspace/data/cifar10/cifar10_data.txt";
   std::string label_file = "/home/zhangshuai20/workspace/data/cifar10/cifar10_labels.txt";
   test_accuracy<NV,Precision::FP32>(FLAGS_model_path, data_file, label_file);
#endif
}


int main(int argc, const char** argv){
    FLAGS_model_path = argv[1];

	Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
#else
int main(int argc, const char** argv) {
    return -1;
}
#endif