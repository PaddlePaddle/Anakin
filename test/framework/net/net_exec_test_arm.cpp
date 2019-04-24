#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include "saber/core/tensor_op.h"
#include <chrono>
#include "debug.h"
#ifdef ENABLE_OP_TIMER
#include "saber/funcs/impl/impl_base.h"
#endif

std::string g_model_path = "";

std::string model_saved_path = g_model_path + ".saved";
int g_batch_size = 1;
int g_warm_up = 0;
int g_epoch = 1;
int g_thread_num = 1;
bool g_random = 0;
int g_instance = 1;
int g_cluster = 0;
bool g_set_archs = false;
ARMArch g_arch = A73;
#ifdef USE_ARM_PLACE

TEST(NetTest, net_execute_base_test) {
    LOG(INFO) << "begin test";
    auto ctx_p = std::make_shared<Context<ARM>>();
    ctx_p->set_run_mode((PowerMode)g_cluster, g_thread_num);
    if (g_set_archs) {
        ctx_p->set_arch(g_arch);
        LOG(INFO) << "arm arc: " << g_arch;
    }
    Graph<ARM, Precision::FP32>* graph = new Graph<ARM, Precision::FP32>();
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

    Net<ARM, Precision::FP32> net_executer(true);
    net_executer.init(*graph, ctx_p);

    for (int j = 0; j < vin_name.size(); ++j) {
        Tensor<ARM>* d_tensor_in_p = net_executer.get_in(vin_name[j]);
        Shape shin = d_tensor_in_p->valid_shape();
        LOG(INFO) << "input tensor size: ";
        LOG(INFO) << "input name: " << vin_name[j];
        for (int k = 0; k < d_tensor_in_p->dims(); ++k) {
            LOG(INFO) << "|---: " << shin[k];
        }
        if (g_random) {
            fill_tensor_rand(*d_tensor_in_p);
        } else {
            fill_tensor_const(*d_tensor_in_p, 1.f);
        }
    }
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
    // do inference
    double to = 0;
    double tmin = 1000000;
    double tmax = 0;
    saber::SaberTimer<ARM> t1;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";

    // warm up
    for (int i = 0; i < g_warm_up; i++) {
        net_executer.prediction();
    }
    for (int i = 0; i < g_epoch; i++) {
        for (int j = 0; j < vin_name.size(); ++j) {
            Tensor<ARM>* d_tensor_in_p = net_executer.get_in(vin_name[j]);
            if (g_random) {
                fill_tensor_rand(*d_tensor_in_p);
            } else {
                fill_tensor_const(*d_tensor_in_p, 1.f);
            }
        }
        t1.clear();
        t1.start(*ctx_p);
        net_executer.prediction();
        t1.end(*ctx_p);
        float tdiff = t1.get_average_ms();
        if (tdiff > tmax) {
            tmax = tdiff;
        }
        if (tdiff < tmin) {
            tmin = tdiff;
        }
        to += tdiff;
        LOG(INFO) << "iter: " << i << ", time: " << tdiff << "ms";
    }
    for (int i = 0; i < out_name.size(); ++i) {
       Tensor<ARM>* vout = net_executer.get_out(out_name[i]);
       write_tensorfile(*vout, out_name[i].c_str());
#ifdef ENABLE_DEBUG
        const float* ptr = vout->data();
        for (int j = 0; j < vout->valid_size(); ++j) {
            printf("%f ", ptr[j]);
            if ((j + 1) % 10 == 0) {
                printf("\n");
            }
        }
        printf("\n");
#endif
        double mean_val = tensor_mean_value_valid(*vout); //tensor_mean(*vout);
        LOG(INFO) << "output mean: " << mean_val;
    }
    LOG(INFO) << "M:" << g_model_path << " th:" << g_thread_num << " batch_size " << g_batch_size << " average time " << to / g_epoch
              << ", min time: " << tmin << "ms, max time: " << tmax << " ms";
#ifdef ENABLE_OP_TIMER
    OpTimer::print_timer(*ctx_p);
    // std::cout << "MC:" << lite_model << " total-ops:" << OpTimer::get_timer("total").ops / FLAGS_epoch << std::endl;
    LOG(INFO) << "MC:" << g_model_path << " total-ops:" << OpTimer::get_timer("total").ops / g_epoch ;
#endif //ENABLE_OP_TIMER
    //    std::string save_g_model_path = g_model_path + std::string(".saved");
    //    status = graph->save(save_g_model_path);
    delete graph;
}

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
    LOG(INFO)<< "usage:";
    LOG(INFO)<< argv[0] << " <anakin model> <num> <warmup_iter> <epoch>";
    LOG(INFO)<< "   lite_model:     path to anakin lite model";
    LOG(INFO)<< "   num:            batchSize default to 1";
    LOG(INFO)<< "   warmup_iter:    warm up iterations default to 10";
    LOG(INFO)<< "   epoch:          time statistic epoch default to 10";
    LOG(INFO)<< "   cluster:        choose which cluster to run, 0: big cores, 1: small cores, 2: all cores, 3: threads not bind to specify cores";
    LOG(INFO)<< "   threads:        set openmp threads";

    if(argc < 2) {
        LOG(ERROR) << "You should fill in the variable lite model at least.";
        return 0;
    }
    g_model_path = std::string(argv[1]);

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
    if (argc > 7) {
        g_set_archs = true;
        if (atoi(argv[7]) > 0) {
            g_arch = (ARMArch)atoi(argv[7]);
        } else {
            g_arch = ARM_UNKOWN;
        }
    }
    if (argc > 8) {
        g_random = atoi(argv[8]);
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
