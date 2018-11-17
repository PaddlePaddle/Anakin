#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "debug.h"


//#define USE_DIEPSE

std::string g_model_path = "/home/liujunjie03/py_anakin/tools/external_converter_v2/output/vggish.anakin.bin";

std::string model_saved_path = g_model_path + ".saved";
int g_batch_size = 1;
int g_warm_up = 10;
int g_epoch = 1000;

#ifdef USE_X86_PLACE

#include <mkl_service.h>
#include "omp.h"
#if 1
TEST(NetTest, net_execute_base_test) {
    Graph<X86, Precision::FP32>* graph = new Graph<X86, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    graph->ResetBatchSize("input_0", g_batch_size);
    graph->Optimize();

    Net<X86, Precision::FP32> net_executer(true);
    net_executer.load_calibrator_config("net_pt_config.txt","cal_file");
    net_executer.init(*graph);
    // get in
//    auto d_tensor_in_p = net_executer.get_in("input_0");
    std::vector<std::string>& vin_name = graph->get_ins();
    for (int j = 0; j < vin_name.size(); ++j) {
        auto d_tensor_in_p = net_executer.get_in(vin_name[j]);
        fill_tensor_const(*d_tensor_in_p, 1.f);
    }


    // do inference
    Context<X86> ctx(0, 0, 0);
    saber::SaberTimer<X86> my_time;

    LOG(WARNING) << "EXECUTER !!!!!!!! ";
	// warm up
	for (int i = 0; i < g_warm_up; i++) {
		net_executer.prediction();
	}

    my_time.start(ctx);
    for (int i = 0; i < g_epoch; i++) {
        net_executer.prediction();
    }
    my_time.end(ctx);
    LOG(INFO)<<"aveage time "<<my_time.get_average_ms()/g_epoch << " ms";
    std::string save_g_model_path = g_model_path + std::string(".saved");
    status = graph->save(save_g_model_path);
    if (!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    if (!graph){
        delete graph;
    }

}
#endif 



int main(int argc, const char** argv){
    if (argc < 2){
        LOG(ERROR)<<"no input!!!";
        return;
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

	Env<X86>::env_init();
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
#else
int main(int argc, const char** argv){

}
#endif
