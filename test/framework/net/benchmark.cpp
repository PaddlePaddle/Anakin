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
#include "framework/operators/ops.h"
#include "saber/core/impl/amd/utils/amd_profiler.h"

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
using Target_H = AMDHX86;
#endif

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
int FLAGS_epoch = 1000;
int FLAGS_device_id = 0;
#endif

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
    std::vector<std::string> models;
    if (FLAGS_model_file == "") {
        getModels(FLAGS_model_dir, models);
    } else {
        models.push_back(FLAGS_model_dir + FLAGS_model_file);
    }
    for (auto iter = models.begin(); iter < models.end(); iter++)
    {
        LOG(WARNING) << "load anakin model file from " << *iter << " ...";
        Graph<Target, Precision::FP32> graph;
        auto status = graph.load(*iter);
        if (!status) {
            LOG(FATAL) << " [ERROR] " << status.info();
        }

        //! get output name
        std::vector<std::string>& vin_name = graph.get_ins();
        LOG(INFO) << "input tensor num: " << vin_name.size();

        //! get output name
        std::vector<std::string>& vout_name = graph.get_outs();
        LOG(INFO) << "output tensor num: " << vout_name.size();

        for (int j = 0; j < vin_name.size(); ++j) {
            LOG(INFO) << "set input " << vin_name[j] << " batchsize to " << FLAGS_num;
            graph.ResetBatchSize(vin_name[j].c_str(), FLAGS_num);
        }
        LOG(INFO) << "optimize the graph";
        graph.Optimize();

        // constructs the executer net
        LOG(INFO) << "create net to execute";
        Net<Target, Precision::FP32> net_executer(graph, true);
        // get in
        LOG(INFO) << "set input";
        for (auto& in : vin_name) {
            auto d_tensor_in_p = net_executer.get_in(in.c_str());
            for (int i = 0; i < d_tensor_in_p->valid_shape().size(); i++) {
                LOG(INFO) << "detect input dims[" << i << "]" << d_tensor_in_p->valid_shape()[i];
            }
            Tensor<Target_H> th(d_tensor_in_p->valid_shape());
            fill_tensor_const(th, 1.f);
            d_tensor_in_p->copy_from(th);
        }
        // do inference
        Context<Target> ctx(FLAGS_device_id, 0, 0);
#if defined(USE_CUDA)
        cudaDeviceSynchronize();
#elif defined(AMD_GPU)
        clFlush(ctx.get_compute_stream());
        clFinish(ctx.get_compute_stream());
#endif
        saber::SaberTimer<Target> my_time;
        LOG(WARNING) << "EXECUTER !!!!!!!! ";

        for (int i = 0; i < FLAGS_warmup_iter; i++) {
            net_executer.prediction();
        }
#if defined(USE_CUDA)
		cudaDeviceSynchronize();
#elif defined(AMD_GPU)
        clFlush(ctx.get_compute_stream());
        clFinish(ctx.get_compute_stream());
#endif

#ifdef ENABLE_OP_TIMER
        net_executer.reset_op_time();
#endif
#ifdef AMD_GPU
        AMDProfiler::start_record();
#endif
        my_time.start(ctx);

        for (int i = 0; i < FLAGS_epoch; i++) {
            for (auto& in : vin_name) {
                auto d_tensor_in_p = net_executer.get_in(in.c_str());
                Tensor<Target_H> th(d_tensor_in_p->valid_shape());
                fill_tensor_const(th, 1.f);
                d_tensor_in_p->copy_from(th);
            }
            net_executer.prediction();
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

        LOG(INFO) << model_name << " batch_size " << FLAGS_num << " average time "<< my_time.get_average_ms() / FLAGS_epoch << " ms";

#ifdef AMD_GPU
       AMDProfiler::stop_record();
       AMDProfiler::pop();
#endif
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
    LOG(INFO)<< "   epoch:          time statistic epoch default to 1000";
    LOG(INFO)<< "   device_id:      select which device to run the model";
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
        FLAGS_device_id = atoi(argv[6]);
    }
#endif

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
