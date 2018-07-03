#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "saber/core/tensor_op.h"
#include <dirent.h> 
#include <sys/stat.h> 
#include <sys/types.h> 
#include <unistd.h>  
#include <fcntl.h>
#include <map>
#ifdef USE_ARM_PLACE
#ifdef USE_GFLAGS
#include <gflags/gflags.h>

DEFINE_string(model_dir, "", "model dir");
DEFINE_string(model_file, "", "model file");
DEFINE_int32(num, 1, "batchSize");
DEFINE_int32(warmup_iter, 10, "warm up iterations");
DEFINE_int32(epoch, 1000, "time statistic epoch");
#else
std::string FLAGS_model_dir;
std::string FLAGS_model_file;
int FLAGS_num = 1;
int FLAGS_warmup_iter = 10;
int FLAGS_epoch = 10;
int FLAGS_threads = 1;
int FLAGS_cluster = 0;
#endif

using Target = ARM;
using Target_H = ARM;

void getModels(std::string path, std::vector<std::string>& files) {
    DIR *dir;
    struct dirent *ptr;
    if ((dir = opendir(path.c_str())) == NULL) {
        perror("Open dri error...");
        exit(1);
    }
    while((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;
        else if (ptr->d_type == 8)//file
            files.push_back(path + "/" + ptr->d_name);
        else if (ptr->d_type == 4) {
            getModels(path + "/" + ptr->d_name, files);
        }
    }
    closedir(dir);
}
TEST(NetTest, net_execute_base_test) {

    std::shared_ptr<Context<ARM>> ctx1 = std::make_shared<Context<ARM>>();

    ctx1->set_run_mode((PowerMode)FLAGS_cluster, FLAGS_threads);

    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    std::vector<std::string> models;
    if (FLAGS_model_file == "") {
        getModels(FLAGS_model_dir, models);
    } else {
        models.push_back(FLAGS_model_dir + FLAGS_model_file);
    }
    for (auto iter = models.begin(); iter < models.end(); iter++)
    {
        LOG(WARNING) << "load anakin model file from " << *iter << " ...";
        Graph<Target, AK_FLOAT, Precision::FP32> graph;   
        auto status = graph.load(*iter);
        if (!status) {
            LOG(FATAL) << " [ERROR] " << status.info();
        }
        LOG(INFO) << "set batchsize to " << FLAGS_num;
        graph.ResetBatchSize("input_0", FLAGS_num);
        LOG(INFO) << "optimize the graph";
        graph.Optimize();
        // constructs the executer net
        LOG(INFO) << "create net to execute";
        Net<Target, AK_FLOAT, Precision::FP32, OpRunType::SYNC> net_executer(graph, ctx1, true);
        // get in
        LOG(INFO) << "get input";
        auto d_tensor_in_p = net_executer.get_in("input_0");
        Tensor4d<Target_H, AK_FLOAT> h_tensor_in;
        auto valid_shape_in = d_tensor_in_p->valid_shape();
        for (int i = 0; i < valid_shape_in.size(); i++) {
            LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
        }
        h_tensor_in.re_alloc(valid_shape_in);
        fill_tensor_host_rand(h_tensor_in, -1.0f,1.0f);
        d_tensor_in_p->copy_from(h_tensor_in);
        // do inference
        Context<Target> ctx(0, 0, 0);
        saber::SaberTimer<Target> my_time;
        LOG(WARNING) << "EXECUTER !!!!!!!! ";
        for (int i = 0; i < FLAGS_warmup_iter; i++) {
            net_executer.prediction();
        }
#ifdef ENABLE_OP_TIMER
        net_executer.reset_op_time();
#endif
        double to = 0;
        double tmin = 1000000;
        double tmax = 0;
        my_time.start(ctx);
        saber::SaberTimer<Target> t1;
        for (int i = 0; i < FLAGS_epoch; i++) {
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
            LOG(INFO) << "iter: " << i << ", time: " << tdiff << "ms";
        }
        my_time.end(ctx);
#ifdef ENABLE_OP_TIMER
        std::vector<float> op_time = net_executer.get_op_time();
        auto exec_funcs = net_executer.get_exec_funcs();
        auto op_param = net_executer.get_op_param();
        for (int i = 0; i <  op_time.size(); i++) {
            LOG(INFO) << "name: " << exec_funcs[i].name << " op_type: " << exec_funcs[i].op_name << " op_param: " << op_param[i] << " time " << op_time[i]/FLAGS_epoch;
        }
        std::map<std::string, float> op_map;
        for (int i = 0; i < op_time.size(); i++) {
            auto it = op_map.find(op_param[i]);
            if (it != op_map.end())
                op_map[op_param[i]] += op_time[i];
            else
                op_map.insert(std::pair<std::string, float>(op_param[i], op_time[i]));
        }
        for (auto it = op_map.begin(); it != op_map.end(); ++it) {
            LOG(INFO)<< it->first << "  " << (it->second) / FLAGS_epoch<< " ms";
        }
#endif
        size_t end = (*iter).find(".anakin.bin");
        size_t start = FLAGS_model_dir.length();
        std::string model_name = (*iter).substr(start, end-start);
        
        LOG(INFO) << model_name << " batch_size " << FLAGS_num << " average time " << to/ FLAGS_epoch << \
            ", min time: " << tmin << "ms, max time: " << tmax << " ms";
       //my_time.get_average_ms() / FLAGS_epoch << " ms";
    }
}
int main(int argc, const char** argv){

    Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);

#ifdef USE_GFLAGS
    google::ParseCommandLineFlags(&argc, &argv, true);
#else 
    LOG(INFO)<< "BenchMark usage:";
    LOG(INFO)<< "   $benchmark <model_dir> <model_file> <num> <warmup_iter> <epoch>";
    LOG(INFO)<< "   model_dir:      model directory";
    LOG(INFO)<< "   model_file:     path to model";
    LOG(INFO)<< "   num:            batchSize default to 1";
    LOG(INFO)<< "   warmup_iter:    warm up iterations default to 10";
    LOG(INFO)<< "   epoch:          time statistic epoch default to 10";
    LOG(INFO)<< "   cluster:        choose which cluster to run, 0: big cores, 1: small cores";
    LOG(INFO)<< "   threads:        set openmp threads";
    if(argc < 3) {
        LOG(ERROR) << "You should fill in the variable model_dir and model_file at least.";
        return 0;
    }
    FLAGS_model_dir = argv[1];
    if(argc > 2) {
        FLAGS_model_file = argv[2];
    }
    if(argc > 3) {
        FLAGS_num = atoi(argv[3]);
    }
    if(argc > 4) {
        FLAGS_warmup_iter = atoi(argv[4]);
    }
    if(argc > 5) {
        FLAGS_epoch = atoi(argv[5]);
    }
    if(argc > 6) {
        FLAGS_cluster = atoi(argv[6]);
        if (FLAGS_cluster < 0) {
            FLAGS_cluster = 0;
        }
        if (FLAGS_cluster > 1) {
            FLAGS_cluster = 1;
        }
    }
    if(argc > 7) {
        FLAGS_threads = atoi(argv[7]);
    }
#endif
    InitTest();
    RUN_ALL_TESTS(argv[0]); 
    return 0;
}

#else
int main(int argc, const char** argv) {
    LOG(INFO) << "this benchmark is only for arm device";
}
#endif