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
#include "debug.h"
#include <string>

#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(std::string, model_dir, "");
DEFINE_GLOBAL(int, num, 1);
DEFINE_GLOBAL(int, channel, 8);
DEFINE_GLOBAL(int, height, 640);
DEFINE_GLOBAL(int, width, 640);
DEFINE_GLOBAL(bool, is_input_shape, false);

#if defined(USE_CUDA)
using Target = NV;
using Target_H = X86;
#elif defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#elif defined(USE_ARM_PLACE)
using Target = ARM;
using Target_H = ARM;
#endif

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

#ifdef USE_CUDA
TEST(NetTest, nv_net_execute_base_test) {
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

        graph.ResetBatchSize("input_0", GLB_num);

        graph.Optimize();
        // constructs the executer net
        Net<NV, AK_FLOAT, Precision::FP32> net_executer(graph, true);
        // get in
        auto d_tensor_in_p = net_executer.get_in("input_0");
        Tensor4d<Target_H, AK_FLOAT> h_tensor_in;
        auto valid_shape_in = d_tensor_in_p->valid_shape();

        for (int i = 0; i < valid_shape_in.size(); i++) {
            LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
        }

        h_tensor_in.re_alloc(valid_shape_in);
//        fill_tensor_host_const(h_tensor_in,1.0f);
        readTensorData(h_tensor_in,"./input_x");
        d_tensor_in_p->copy_from(h_tensor_in);
        int warmup_iter = 10;
        int epoch = 1000;
        // do inference
        Context<NV> ctx(0, 0, 0);
        saber::SaberTimer<NV> my_time;
        LOG(WARNING) << "EXECUTER !!!!!!!! ";
        net_executer.prediction();
        int count=0;
        for (auto out:net_executer.get_out_list()){
            LOG(INFO)<<"out "<<count;
            record_dev_tensorfile(out,(std::string("dev_")+std::to_string(count++)).c_str());
        }



    LOG(INFO) << *iter << " aveage time " << my_time.get_average_ms() / epoch << " ms";
#endif
    }
}
#endif

int main(int argc, const char** argv) {

    Env<Target>::env_init();

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
    }
    GLB_model_dir = std::string(argv[1]);
    if(argc==3){
        GLB_num = atoi(argv[2]);
    }

    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
