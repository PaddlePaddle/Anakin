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
#include "paddle_api.h"s
#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(std::string, model_dir, "");
DEFINE_GLOBAL(int, num, 1);
DEFINE_GLOBAL(int, channel, 8);
DEFINE_GLOBAL(int, height, 640);
DEFINE_GLOBAL(int, width, 640);
DEFINE_GLOBAL(bool, is_input_shape, false);
#ifdef USE_CUDA
void getModels(std::string path, std::vector<std::string>& files)
{
    DIR *dir;
    struct dirent *ptr;
    if((dir=opendir(path.c_str()))==NULL){
        perror("Open dri error...");
        exit(1);
    }
    while((ptr=readdir(dir))!=NULL){
        if(strcmp(ptr->d_name,".")==0||strcmp(ptr->d_name,"..")==0)
            continue;
        else if(ptr->d_type==8)//file
            files.push_back(path+"/"+ptr->d_name);
        else if(ptr->d_type==4){
            //files.push_back(ptr->d_name);//dir
            getModels(path+"/"+ptr->d_name,files);
        }
    }
    closedir(dir);
}


TEST(NetTest, net_execute_base_test) {
    std::vector<std::string> models;
    getModels(GLB_model_dir, models);
    for (auto iter = models.begin(); iter < models.end(); iter++)
    {
        AnakinEngine<NV, AK_FLOAT, Precision::FP32> anakin_engine;
        LOG(WARNING) << "load anakin model file from " << *iter << " ...";
        std::vector<int> shape{GLB_num, GLB_channel, GLB_height, GLB_width};
        //anakin_engine.Build(*iter, shape);
        anakin_engine.Build(*iter);

        printf("Args = %d %d %d %d\n",GLB_num, GLB_channel, GLB_height, GLB_width);
        //fill input
        Tensor4d<X86, AK_FLOAT> h_tensor_in;
        h_tensor_in.re_alloc({GLB_num, GLB_channel, GLB_height, GLB_width});
        fill_tensor_host_rand(h_tensor_in, -1.0f,1.0f);

        anakin_engine.SetInputFromCPU("input_0", h_tensor_in.data(), h_tensor_in.valid_size());

        int warmup_iter = 10;
        int epoch = 1000;
        // do inference
        Context<NV> ctx(0, 0, 0);
        saber::SaberTimer<NV> my_time;
        LOG(WARNING) << "EXECUTER !!!!!!!! ";
        for (int i = 0; i < warmup_iter; i++) {
            anakin_engine.Execute();
        }
        my_time.start(ctx);
        //auto start = std::chrono::system_clock::now();
        for (int i = 0; i < epoch; i++) {
            anakin_engine.Execute();
        }
        my_time.end(ctx);
        LOG(INFO) << *iter << " aveage time "<< my_time.get_average_ms() / epoch << " ms";            
    }
}

int main(int argc, const char** argv){

    Env<NV>::env_init();
    // initial logger
    LOG(INFO)<<"argc"<<argc;
    if (argc < 1) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/model_test\n \
            anakin_models\n \
            num\n \
            channel\n \
            height\n \
            width\n ";
        exit(0);
    } else if (argc == 2){
        GLB_model_dir = std::string(argv[1]);
        GLB_is_input_shape = false;
    } else if (argc == 3){
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

#else
int main(int argc, char** argv) {
        return 0;
}

#endif
