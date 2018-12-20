#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "debug.h"
#include <fstream>

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
    } else {
        LOG(FATAL) << path << "can not be opened";
    }
}

//#define USE_DIEPSE

std::string g_model_path = "/path/to/your/anakin_model";

std::string model_saved_path = g_model_path + ".saved";
int g_batch_size = 1;
int g_warm_up = 10;
int g_epoch = 1000;
int g_device_id = 0;

#ifdef USE_X86_PLACE
#ifdef TEST_FAST_RCNN

TEST(NetTest, net_execute_base_test) {
    std::string image_file = "/home/chengyujuan/baidu/sys-hic-gpu/Anakin-2.0_fast/fast_rcnn_data/debug_data/feed_image.txt";
    std::string image_info_file = "/home/chengyujuan/baidu/sys-hic-gpu/Anakin-2.0_fast/fast_rcnn_data/fast-rcnn/feed_im_inof.txt";
    Graph<X86, Precision::FP32>* graph = new Graph<X86, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if (!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    // reshape the input_0 's shape for graph model
	graph->ResetBatchSize("input_0", g_batch_size);

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
    Net<X86, Precision::FP32> net_executer(true);

    net_executer.init(*graph);
    // get in
    auto d_image = net_executer.get_in("input_0");
    auto d_image_info = net_executer.get_in("input_1");

    auto image_shape = d_image->valid_shape();
    auto image_info_shape = d_image_info->valid_shape();
    //d_image->re_alloc(image_shape);
    //d_image_info->re_alloc(image_info_shape);
    for (int i = 0; i < image_shape.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << image_shape[i];
    }
    for (int i = 0; i < image_info_shape.size(); i++) {
        LOG(INFO) << "detect input_1 dims[" << i << "]" << image_info_shape[i];
    }

    
    float* image_data = (float*)(d_image->mutable_data());
    float* image_info_data = (float*)(d_image_info->mutable_data());
    read_tensor_from_file(image_data, d_image->valid_size(), image_file.c_str());
    read_tensor_from_file(image_info_data, d_image_info->valid_size(), image_info_file.c_str());

    //int g_epoch = 1000;
    //int g_warm_up=10;
    // do inference
    Context<X86> ctx(g_device_id, 0, 0);
    saber::SaberTimer<X86> my_time;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";
	// warm up
	for (int i = 0; i < g_warm_up; i++) {
		net_executer.prediction();
	}

#ifdef ENABLE_OP_TIMER
    net_executer.reset_op_time();
#endif

    my_time.start(ctx);

    for (int i = 0; i < g_epoch; i++) {
        net_executer.prediction();
    }

    my_time.end(ctx);
#ifdef ENABLE_OP_TIMER
    net_executer.print_and_reset_optime_summary(g_epoch);
#endif

    LOG(INFO)<<"aveage time "<<my_time.get_average_ms()/g_epoch << " ms";
    write_tensorfile(*net_executer.get_out_list()[0],"cpu_generate_proposals_0.txt");
    write_tensorfile(*net_executer.get_out_list()[1],"cpu_bbox_pred.tmp_1.txt");
    write_tensorfile(*net_executer.get_out_list()[2],"cpu_softmax_0.tmp_0.txt");
	////} // inner scope over

	//LOG(ERROR) << "inner net exe over !";
    //for (auto x:net_executer.get_out_list()){
    //    print_tensor(*x);
    //}

    // save the optimized model to disk.
    std::string save_g_model_path = g_model_path + std::string(".saved");
    status = graph->save(save_g_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }
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
        g_batch_size = atoi(argv[2]);
    }
    if (argc > 3) {
        g_warm_up = atoi(argv[3]);
    }
    if (argc > 4) {
        g_epoch = atoi(argv[4]);
    }
    if (argc > 5) {
        g_device_id = atoi(argv[5]);
    }
#ifdef USE_X86_PLACE
    //TargetWrapper<X86>::set_device(g_device_id);
    Env<X86>::env_init();
#endif
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
