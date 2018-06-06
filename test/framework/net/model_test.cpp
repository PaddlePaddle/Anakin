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
#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(std::string, model_dir, "");
DEFINE_GLOBAL(int, num, 1);
DEFINE_GLOBAL(int, channel, 8);
DEFINE_GLOBAL(int, height, 640);
DEFINE_GLOBAL(int, width, 640);
DEFINE_GLOBAL(bool, is_input_shape, false);
void getModels(std::string path, std::vector<std::string>& files) {
    DIR* dir= nullptr;
    struct dirent* ptr;

    if ((dir = opendir(path.c_str())) == NULL) {
        perror("Open dri error...");
        exit(1);
    }

    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
            continue;
        } else if (ptr->d_type == 8) { //file
            files.push_back(path + "/" + ptr->d_name);
        } else if (ptr->d_type == 4) {
            //files.push_back(ptr->d_name);//dir
            getModels(path + "/" + ptr->d_name, files);
        }
    }

    closedir(dir);
}
TEST(NetTest, net_execute_base_test) {
    std::vector<std::string> models;
    getModels(GLB_model_dir, models);

    for (auto iter = models.begin(); iter < models.end(); iter++) {
        LOG(WARNING) << "load anakin model file from " << *iter << " ...";
#if 1
        Graph<NV, AK_FLOAT, Precision::FP32> graph;
        auto status = graph.load(*iter);

        if (!status) {
            LOG(FATAL) << " [ERROR] " << status.info();
        }

        if (GLB_is_input_shape) {
            graph.Reshape("input_0", {GLB_num, GLB_channel, GLB_height, GLB_width});
        } else {
            graph.ResetBatchSize("input_0", GLB_num);
        }

        graph.Optimize();
        // constructs the executer net
        Net<NV, AK_FLOAT, Precision::FP32> net_executer(graph, true);
        // get in
        auto d_tensor_in_p = net_executer.get_in("input_0");
        Tensor4d<X86, AK_FLOAT> h_tensor_in;
        auto valid_shape_in = d_tensor_in_p->valid_shape();

        for (int i = 0; i < valid_shape_in.size(); i++) {
            LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
        }

        h_tensor_in.re_alloc(valid_shape_in);
        fill_tensor_host_rand(h_tensor_in, -1.0f, 1.0f);
        d_tensor_in_p->copy_from(h_tensor_in);
        int warmup_iter = 10;
        int epoch = 1000;
        // do inference
        Context<NV> ctx(0, 0, 0);
        saber::SaberTimer<NV> my_time;
        LOG(WARNING) << "EXECUTER !!!!!!!! ";

        for (int i = 0; i < warmup_iter; i++) {
            net_executer.prediction();
        }

#ifdef ENABLE_OP_TIMER
        net_executer.reset_op_time();
#endif
        my_time.start(ctx);

        //auto start = std::chrono::system_clock::now();
        for (int i = 0; i < epoch; i++) {
            //DLOG(ERROR) << " epoch(" << i << "/" << epoch << ") ";
            net_executer.prediction();
        }

        my_time.end(ctx);
#ifdef ENABLE_OP_TIMER
        std::vector<float> op_time = net_executer.get_op_time();
        auto exec_funcs = net_executer.get_exec_funcs();
        auto op_param = net_executer.get_op_param();

        for (int i = 0; i <  op_time.size(); i++) {
            LOG(INFO) << "name: " << exec_funcs[i].name << " op_type: " << exec_funcs[i].op_name <<
                      " op_param: " << op_param[i] << " time " << op_time[i] / epoch;
        }

        std::map<std::string, float> op_map;

        for (int i = 0; i < op_time.size(); i++) {
            auto it = op_map.find(op_param[i]);

            if (it != op_map.end()) {
                op_map[op_param[i]] += op_time[i];
            } else {
                op_map.insert(std::pair<std::string, float>(op_param[i], op_time[i]));
            }
        }

        for (auto it = op_map.begin(); it != op_map.end(); ++it) {
            LOG(INFO) << it->first << "  " << (it->second) / epoch << " ms";
        }

#endif
        LOG(INFO) << *iter << " aveage time " << my_time.get_average_ms() / epoch << " ms";
        // save the optimized model to disk.
        //        std::string save_model_path = GLB_model_dir + std::string("opt.saved");
        //        status = graph.save(save_model_path);
        //        if (!status ) {
        //            LOG(FATAL) << " [ERROR] " << status.info();
        //        }
#endif
    }
}
int main(int argc, const char** argv) {
    // initial logger
    LOG(INFO) << "argc " << argc;

    if (argc < 1) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/model_test\n \
            anakin_models\n \
            num\n \
            channel\n \
            height\n \
            width\n ";
        exit(0);
    } else if (argc == 2) {
        GLB_model_dir = std::string(argv[1]);
        GLB_is_input_shape = false;
    } else if (argc == 3) {
        GLB_model_dir = std::string(argv[1]);
        GLB_num = atoi(argv[2]);
        GLB_is_input_shape = false;
    } else {
        GLB_model_dir = std::string(argv[1]);
        GLB_num = atoi(argv[2]);
        GLB_channel = atoi(argv[3]);
        GLB_height = atoi(argv[4]);
        GLB_width = atoi(argv[5]);
        GLB_is_input_shape = true;
    }

    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
