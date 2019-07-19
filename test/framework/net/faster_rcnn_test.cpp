#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "debug.h"
#include <fstream>
#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

void read_tensor_from_file(float* data, int length, const char* path) {
    std::fstream fs(path);
    int i = 0;
    if (fs.is_open()) {
        std::string str;
        while(true) {
            std::getline(fs, str);
            std::size_t found = str.find(" ");
            if (found != std::string::npos) {
                std::cout << "first 'needle' found at: " << found << '\n';
                break;
            }
            data[i++] = (atof)(str.c_str());
        }
        fs.close();
    }
}
#if defined(USE_OPENCV) && defined(USE_CUDA)
void fill_image_data(const cv::Mat& img, float * gpu_data, float* gpu_info, int batch){
	int elem_num = img.channels() * img.rows * img.cols;
	float * cpu_data = new float[elem_num];
	// eliminate the padding added by opencv: NHWC
	int idx = 0;
	float scale = 1.0f / 255;
	for(int c = 0; c < img.channels(); c++){
		for(int h = 0; h < img.rows; h++){
			for(int w = 0; w < img.cols; w++)
				cpu_data[idx++] = img.data[h * img.step + w * img.channels() + c] * scale;
		}
	}
    float* cpu_info = new float[3];
    cpu_info[0] = float(img.rows);
    cpu_info[1] = float(img.cols);
    cpu_info[2] = 1.f;
	// TODO: use anakin API
	for (int i = 0; i < batch; i++) {
	    cudaMemcpy(gpu_data + i * elem_num, cpu_data, elem_num* sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(gpu_info + i * 3, cpu_info, 3* sizeof(float), cudaMemcpyHostToDevice);
    }

	delete[]  cpu_data;
	delete[]  cpu_info;
}
#endif

//#define USE_DIEPSE

std::string g_model_path = "/path/to/your/anakin_model";

std::string model_saved_path = g_model_path + ".saved";
int g_batch_size = 1;
int g_warm_up = 10;
int g_epoch = 1000;
int g_device_id = 0;
int g_start = 0;
int g_end = 0;
std::string g_image_list = "";
//#define TEST_FAST_RCNN

#ifdef TEST_FAST_RCNN
#ifdef USE_CUDA

TEST(NetTest, net_execute_base_test) {

    std::ifstream ifs(g_image_list.c_str(), std::ifstream::in);
    CHECK(ifs.is_open()) << g_image_list << " can not be opened";
    std::vector<std::string> file_list;
    while (ifs.good()) {
        std::string new_file;
        std::getline(ifs, new_file);
        file_list.push_back(new_file);
    }

    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    // reshape the input_0 's shape for graph model
	graph->ResetBatchSize("input_0", g_batch_size);

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
    Net<NV, Precision::FP32> net_executer(true);

    //net_executer.load_calibrator_config("net_pt_config.txt","cal_file");
    net_executer.init(*graph);
    // get in
    auto d_image = net_executer.get_in("input_0");
    auto d_image_info = net_executer.get_in("input_1");
    Tensor4d<NVHX86> h_image;
    Tensor4d<NVHX86> h_image_info;

    auto image_shape = d_image->valid_shape();
    auto image_info_shape = d_image_info->valid_shape();
    for (int i = 0; i < image_shape.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << image_shape[i];
    }
    for (int i = 0; i < image_info_shape.size(); i++) {
        LOG(INFO) << "detect input_1 dims[" << i << "]" << image_info_shape[i];
    }

    Context<NV> ctx(g_device_id, 0, 0);
    saber::SaberTimer<NV> my_time;
#ifdef USE_OPENCV
    for (int i = g_start; i < file_list.size() && i < g_end; i++) {
        int img_id = 0;
        cv::Mat img = cv::imread(file_list[img_id], cv::IMREAD_COLOR);
        if (img.empty()) {
            LOG(FATAL) << "load image " << file_list[img_id] << " failed";
        }
        Shape image_shape({g_batch_size, img.channels(), img.rows, img.cols}, Layout_NCHW);
        Shape info_shape({g_batch_size, 3, 1, 1}, Layout_NCHW);
        d_image->reshape(image_shape);
        d_image_info->reshape(info_shape);
        float* gpu_image = (float*)d_image->mutable_data();
        float* gpu_image_info = (float*)d_image_info->mutable_data();
        fill_image_data(img, gpu_image, gpu_image_info, g_batch_size);
        cudaDeviceSynchronize();
        //write_tensorfile(*d_image, "image.txt");
        //write_tensorfile(*d_image_info, "image_info.txt");
        net_executer.prediction();
        if (i - g_start == g_warm_up) {
#ifdef ENABLE_OP_TIMER
            net_executer.reset_op_time();
#endif
            my_time.start(ctx);
        }
    }
#endif
    cudaDeviceSynchronize();
    my_time.end(ctx);
#ifdef ENABLE_OP_TIMER
    net_executer.print_and_reset_optime_summary(g_epoch);
#endif

    LOG(INFO)<<"aveage time "<<my_time.get_average_ms()/g_epoch << " ms";
    write_tensorfile(*net_executer.get_out_list()[0],"generate_proposals_0.txt");

    if (!graph) {
        delete graph;
    }
}
#endif
#endif


int main(int argc, const char** argv){
    if (argc < 2){
        LOG(ERROR)<<"no input!!!";
        return -1;
    }
    if (argc > 1) {
        g_model_path = std::string(argv[1]);
    }
    if (argc > 2) {
        g_image_list = std::string(argv[2]);
    }
    if (argc > 3) {
        g_batch_size = atoi(argv[3]);
    }
    if (argc > 4) {
        g_warm_up = atoi(argv[4]);
    }
    if (argc > 5) {
        g_epoch = atoi(argv[5]);
    }
    if (argc > 6) {
        g_device_id = atoi(argv[6]);
    }
    if (argc > 7) {
        g_start = atoi(argv[7]);
    }
    if (argc > 8) {
        g_end = atoi(argv[8]);
    }

#ifdef USE_CUDA
    TargetWrapper<NV>::set_device(g_device_id);
    Env<NV>::env_init();
#endif
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);
	return 0;
}
