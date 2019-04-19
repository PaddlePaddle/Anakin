#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "debug.h"


//#define USE_DIEPSE

std::string g_model_path = "";

std::string model_saved_path = g_model_path + ".saved";
int g_batch_size = 1;
int g_warm_up = 0;
int g_epoch = 1;
int g_thread_num = 1;
bool g_random = 0;
int g_instance = 1;
int g_change_batch = 0;
int g_auto_config_layout = 0;
#define USE_FROZEN_INT8 0

#ifdef USE_X86_PLACE

#include "mkl_service.h"
#include "omp.h"
#if 1
void  instance_run() {

    if (g_thread_num != 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(g_thread_num);
        mkl_set_num_threads(g_thread_num);
    } else {
        LOG(INFO) << "use all core!!";
    }

    LOG(INFO) << "set thread = " << g_thread_num << " , " << mkl_get_max_threads() << "," <<
              omp_get_max_threads();

#if USE_FROZEN_INT8
    Graph<X86, Precision::INT8>* graph = new Graph<X86, Precision::INT8>();
#else
    Graph<X86, Precision::FP32>* graph = new Graph<X86, Precision::FP32>();
#endif
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);

    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
#if USE_FROZEN_INT8

#else
    graph->load_calibrator_config("net_pt_config", "cal_file");
    graph->load_layout_config("model_layout_config");
#endif
    //    graph->Reshape("input_0",Shape({1,3,400,600},Layout_NCHW));
    std::vector<std::string>& vin_name = graph->get_ins();

    for (int j = 0; j < vin_name.size(); ++j) {
        graph->ResetBatchSize("input_0", g_batch_size);
    }

#if USE_FROZEN_INT8
    graph->Optimize(false);
#else
    graph->Optimize();
#endif

#if USE_FROZEN_INT8
    Net<X86, Precision::INT8> net_executer(true);
#else
    Net<X86, Precision::FP32> net_executer(true);
#endif
    if (g_auto_config_layout){
        LOG(INFO) << "===================auto_config_layout====================";
        net_executer.init(*graph,true);
    }else {
//        net_executer.load_x86_layout_config("layout_config_me.txt");
        net_executer.init(*graph);
    }
    // get in
    std::vector<std::vector<int>> seq_offset={{0,g_batch_size}};
    srand(12345);

    for (int j = 0; j < vin_name.size(); ++j) {
        Tensor<X86>* d_tensor_in_p = net_executer.get_in(vin_name[j]);
        //        d_tensor_in_p->reshape(Shape({1,3,400,600},Layout_NCHW));
        LOG(INFO) << "input name: " << vin_name[j] << " , " << d_tensor_in_p->valid_shape();
        d_tensor_in_p->set_seq_offset(seq_offset);
        if (g_random) {
            fill_tensor_rand(*d_tensor_in_p);
        } else {
            fill_tensor_const(*d_tensor_in_p, 1.f);
        }
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

    int real_batch=1;
    for (int i = 0; i < g_epoch; i++) {
        if (g_change_batch > 0) {
            real_batch = real_batch < g_batch_size ? real_batch + 1 : 1;
            for (int j = 0; j < vin_name.size(); ++j) {
                Tensor<X86>* d_tensor_in_p = net_executer.get_in(vin_name[j]);
                Shape old_shape=d_tensor_in_p->valid_shape();
                old_shape.set_num(real_batch);
                d_tensor_in_p->reshape(old_shape);
                if (g_random) {
                    fill_tensor_rand(*d_tensor_in_p);
                } else {
                    fill_tensor_const(*d_tensor_in_p, 1.f);
                }
            }
        }

        net_executer.prediction();
    }

    my_time.end(ctx);
    LOG(INFO) << "g_auto_config_layout:" << g_auto_config_layout;
    LOG(INFO) << "average time " << my_time.get_average_ms() / g_epoch << " ms";

    std::vector<std::string>& out_name = graph->get_outs();

    for (int j = 0; j < out_name.size(); ++j) {
        LOG(INFO) << "output tensor : " << out_name[j]<<","<<net_executer.get_out(out_name[j])->valid_shape();
        write_tensorfile(*net_executer.get_out(out_name[j]), out_name[j].c_str());
    }

#ifdef ENABLE_OP_TIMER
    net_executer.print_and_reset_optime_summary(g_warm_up + g_epoch);
#endif

    //    std::string save_g_model_path = g_model_path + std::string(".saved");
    //    status = graph->save(save_g_model_path);
    delete graph;
}
#endif

void multi_instance_run(){
    std::vector<std::unique_ptr<std::thread>> instances_vec;
    for (int i = 0; i < g_instance; ++i) {
        instances_vec.emplace_back(
                new std::thread(&instance_run));
    }
    for (int i = 0; i < g_instance; ++i) {
        instances_vec[i]->join();
    }

}

#if 0
void  net_execute_base_test_int8() {

    if (g_thread_num != 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(g_thread_num);
        mkl_set_num_threads(g_thread_num);
    } else {
        LOG(INFO) << "use all core!!";
    }

    LOG(INFO) << "set thread = " << g_thread_num << " , " << mkl_get_max_threads() << "," <<
              omp_get_max_threads();

    Graph<X86, Precision::INT8>* graph = new Graph<X86, Precision::INT8>();

    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    graph->load_calibrator_config("net_pt_config.txt", "cal_file");
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    //    graph->Reshape("input_0",Shape({1,3,400,600},Layout_NCHW));
    std::vector<std::string>& vin_name = graph->get_ins();

    for (int j = 0; j < vin_name.size(); ++j) {
        graph->ResetBatchSize("input_0", g_batch_size);
    }

    graph->Optimize();

    Net<X86, Precision::INT8> net_executer(true);
    net_executer.load_x86_layout_config("layout_config_me.txt");
    net_executer.init(*graph);
    // get in

    srand(12345);

    for (int j = 0; j < vin_name.size(); ++j) {
        Tensor<X86>* d_tensor_in_p = net_executer.get_in(vin_name[j]);
        //        d_tensor_in_p->reshape(Shape({1,3,400,600},Layout_NCHW));
                LOG(INFO) << "input name: " << vin_name[j] << " , " << d_tensor_in_p->valid_shape();

        if (g_random) {
            fill_tensor_rand(*d_tensor_in_p);
        } else {
            fill_tensor_const(*d_tensor_in_p, 1.f);
        }
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
    LOG(INFO) << "average time " << my_time.get_average_ms() / g_epoch << " ms";

    std::vector<std::string>& out_name = graph->get_outs();

    for (int j = 0; j < out_name.size(); ++j) {
        LOG(INFO) << "output tensor : " << out_name[j]<<","<<net_executer.get_out(out_name[j])->valid_shape();
        write_tensorfile(*net_executer.get_out(out_name[j]), out_name[j].c_str());
    }

#ifdef ENABLE_OP_TIMER
    net_executer.print_and_reset_optime_summary(g_warm_up + g_epoch);
#endif

    //    std::string save_g_model_path = g_model_path + std::string(".saved");
    //    status = graph->save(save_g_model_path);
}
#endif

/**
 * g_model_path 模型地址
 * g_batch_size batch大小,默认1
 * g_warm_up 预热次数,默认0
 * g_epoch 计时次数,默认1
 * g_thread_num 用到的线程数,默认1
 * g_random 是否是随机数输入,默认是,0代表常量输入
 * @param argc
 * @param argv
 * @return
 */

int main(int argc, const char** argv) {
//    LOG(INFO)<<"kmp_get_affinity_max_proc = "<<omp_get_num_devices();
//    LOG(INFO)<<"kmp_get_affinity_max_proc = "<<omp_get_proc_bind();
    if (argc < 2) {
        LOG(ERROR) << "no input!!!";
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
        g_thread_num = atoi(argv[5]);
    }

    if (argc > 6) {
        g_random = atoi(argv[6]);
    }

    if (argc > 7) {
        g_auto_config_layout = atoi(argv[7]);
    }

    if (argc > 8) {
        g_instance = atoi(argv[8]);
    }

    if (argc > 9) {
        g_change_batch = atoi(argv[9]);
    }



    Env<X86>::env_init();
    // initial logger
    logger::init(argv[0]);

    multi_instance_run();

    return 0;
}
#else
int main(int argc, const char** argv) {
    return 0;
}
#endif
