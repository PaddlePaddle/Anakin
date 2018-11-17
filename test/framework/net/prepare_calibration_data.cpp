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
int FLAGS_img_num = 1;
int FLAGS_height = 1;
int FLAGS_width = 1;
int FLAGS_channel = 3;
int FLAGS_batch_size = 100;
std::string FLAGS_output_prefix;


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
 

void prepare_batch_data(int img_size,
                   int batch_size,
                   int img_num,
                   std::string data_file) {
    std::ifstream ifs_data;
    float mean[3] = {125.307, 122.95, 113.865};
    float scale = 1.f;
    ifs_data.open(data_file);
    CHECK(ifs_data.is_open()) << "data file can not be opened";

    Tensor4d<Target_H> h_tensor_in;
    auto valid_shape_in = std::vector<int>{batch_size, img_size, 1, 1};
    std::string out_file;
    std::ofstream out_car;
    h_tensor_in.re_alloc(valid_shape_in);
    auto h_data = (float*) h_tensor_in.mutable_data();
    int img_id = 0;
    while(img_id < img_num) {
        if (img_id % batch_size == 0) {
             out_car.close();
            char name[100];
            sprintf(name, "Batch%d", img_id/batch_size);
            out_file = FLAGS_output_prefix + name;
            out_car.open(out_file, std::ofstream::out|std::ofstream::binary);
            if( ! out_car.is_open() ){
                LOG(ERROR) << " failed to open out car file: " << out_file;
                return -1;
            }
            int num = std::min(img_num - img_id, batch_size);
            out_car.write((const char*)&num, sizeof(int));
            out_car.write((const char*)(&FLAGS_channel), sizeof(int));
            out_car.write((const char*)(&FLAGS_height), sizeof(int));
            out_car.write((const char*)(&FLAGS_width), sizeof(int));
            int i = 0;
            std::string str;
            while(i < num * img_size) {
                getline(ifs_data, str);
                if (str == "") {
                   break;
                }
                auto mean_id = (i / img_size) % 3;
                h_data[i] = scale * (atof(str.c_str()) - mean[mean_id]);
                i++;
            }
            if (i == 0) {
               break;
            }
            out_car.write((const char*)(h_data), num * img_size * sizeof(float));
            out_car.close();
            img_id += num;
        }
    }
}

TEST(NetTest, net_execute_base_test) {
    std::string FLAGS_data_file = "/home/zhangshuai20/workspace/data/cifar10/cifar10_data.txt";

    prepare_batch_data(FLAGS_channel * FLAGS_height* FLAGS_width,
                   FLAGS_batch_size,
                   FLAGS_img_num,
                   FLAGS_data_file);
}


int main(int argc, const char** argv){
    FLAGS_img_num = atoi(argv[1]);
    FLAGS_batch_size = atoi(argv[2]);
    FLAGS_channel = atoi(argv[3]);
    FLAGS_height = atoi(argv[4]);
    FLAGS_width = atoi(argv[5]);

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