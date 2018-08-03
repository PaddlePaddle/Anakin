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

#ifdef USE_X86_PLACE
#include <mkl_service.h>
#ifdef USE_OPENMP
#include "omp.h"
#endif
using Target = X86;
using Target_H = X86;

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
int FLAGS_warmup_iter = 1;
int FLAGS_epoch = 2000;
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
#ifdef USE_OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    mkl_set_num_threads(1);
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

                LOG(INFO) << "optimize the graph";
        graph.Optimize();

        //! get output name
        std::vector<std::string>& vin_name = graph.get_ins();
                LOG(INFO) << "input size: " << vin_name.size();
                LOG(INFO) << "set batchsize to " << FLAGS_num;
        for (int j = 0; j < vin_name.size(); ++j) {
                    LOG(INFO) << "input name: " << vin_name[j];
            graph.ResetBatchSize(vin_name[j], FLAGS_num);
        }
        //! get output name
        std::vector<std::string>& vout_name = graph.get_outs();
                LOG(INFO) << "output size: " << vout_name.size();
        for (int j = 0; j < vout_name.size(); ++j) {
                    LOG(INFO) << "output name: " << vout_name[j];
        }

        // constructs the executer net
                LOG(INFO) << "create net to execute";
        Net<Target, AK_FLOAT, Precision::FP32> net_executer(graph, true);//, OpRunType::SYNC
        // get in
                LOG(INFO) << "get input";
        for (int j = 0; j < vin_name.size(); ++j) {
            Tensor<Target, AK_FLOAT>* d_tensor_in_p = net_executer.get_in(vin_name[j]);
            Tensor4d<Target_H, AK_FLOAT> h_tensor_in;
            auto valid_shape_in = d_tensor_in_p->valid_shape();
                    LOG(INFO) << "input name: " << vin_name[j];
            for (int i = 0; i < valid_shape_in.size(); i++) {
                        LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
            }
            //h_tensor_in.re_alloc(valid_shape_in);
            //fill_tensor_host_rand(h_tensor_in, -1.0f,1.0f);
            //d_tensor_in_p->copy_from(h_tensor_in);
            fill_tensor_host_const(*d_tensor_in_p, 1.f);
        }

        // do inference
        Context<Target> ctx(0, 0, 0);
        saber::SaberTimer<Target> my_time;
                LOG(WARNING) << "EXECUTER !!!!!!!! ";
        for (int i = 0; i < FLAGS_warmup_iter; i++) {
            for (int j = 0; j < vin_name.size(); ++j) {
                auto d_tensor_in_p = net_executer.get_in(vin_name[j]);
                fill_tensor_host_const(*d_tensor_in_p, 1.f);
            }
            net_executer.prediction();
        }
#ifdef ENABLE_OP_TIMER
        net_executer.reset_op_time();
#endif
        my_time.start(ctx);
        //auto start = std::chrono::system_clock::now();
        for (int i = 0; i < FLAGS_epoch; i++) {
            //DLOG(ERROR) << " epoch(" << i << "/" << epoch << ") ";
            for (int j = 0; j < vin_name.size(); ++j) {
                auto d_tensor_in_p = net_executer.get_in(vin_name[j]);
                fill_tensor_host_const(*d_tensor_in_p, 1.f);
            }

            net_executer.prediction();

#define LOG_OUTPUT
#ifdef LOG_OUTPUT
            std::vector<Tensor4d<Target, AK_FLOAT>*> vout;
            for (auto& it : vout_name) {
                vout.push_back(net_executer.get_out(it));
            }
            Tensor4d<Target, AK_FLOAT>* tensor_out = vout[0];
                    LOG(INFO) << "output size: " << vout.size();

                    LOG(INFO) << "extract data: size: " << tensor_out->valid_size() << \
            ", width=" << tensor_out->width() << ", height=" << tensor_out->height();
            Tensor4d<Target_H, AK_FLOAT> out_h;
            out_h.re_alloc(tensor_out->valid_shape());
            out_h.copy_from(*tensor_out);
            const float* ptr_out = out_h.data();
            double mean_val = 0.0;
            bool flag_pass = true;
            for (int j = 0; j < tensor_out->valid_size(); j++) {
                if (fabs(ptr_out[j] - 0.708581f) > 1e-5f) {
                    flag_pass = false;
                }
                printf("out = %.8f ", ptr_out[j]);
                mean_val += ptr_out[j];
                if ((j + 1) % 10 == 0) {
                    printf("\n");
                }
            }
            printf("\n");
            if (!flag_pass) {
                        LOG(FATAL) << "error in loop " << i;
            }
                    LOG(INFO) << "output mean val: " << mean_val / tensor_out->valid_size();
#endif
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

        auto status1 = graph.save("map.anakin.bin");
    }
}
int main(int argc, const char** argv){

    //Env<Target>::env_init();

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
#endif
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
#else
int main(int argc, const char** argv){
    return 0;
}
#endif
