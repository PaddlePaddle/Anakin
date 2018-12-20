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
#include <fstream>
#include <thread>


#if  defined(NVIDIA_GPU)
using Target = NV;
using Target_H = NVHX86;
#elif  defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#include "mkl_service.h"
#elif  defined(AMD_GPU)
using Target = AMD;
using Target_H = AMDHX86;
#endif



std::string FLAGS_data_file;
std::string FLAGS_model_file;
int FLAGS_num = 1;
int FLAGS_batch_size = 1;
int FLAGS_thread_num = 1;


std::vector<std::string> string_split(std::string in_str, std::string delimiter) {
    std::vector<std::string> seq;
    int found = in_str.find(delimiter);
    int pre_found = -1;

    while (found != std::string::npos) {
        if (pre_found == -1) {
            seq.push_back(in_str.substr(0, found));
        } else {
            seq.push_back(in_str.substr(pre_found + delimiter.length(),
                                        found - delimiter.length() - pre_found));
        }

        pre_found = found;
        found = in_str.find(delimiter, pre_found + delimiter.length());
    }

    seq.push_back(in_str.substr(pre_found + 1, in_str.length() - (pre_found + 1)));
    return seq;
}
std::vector<std::string> string_split(std::string in_str, std::vector<std::string>& delimiter) {
    std::vector<std::string> in;
    std::vector<std::string> out;
    out.push_back(in_str);

    for (auto del : delimiter) {
        in = out;
        out.clear();

        for (auto s : in) {
            auto out_s = string_split(s, del);

            for (auto o : out_s) {
                out.push_back(o);
            }
        }
    }

    return out;
}

class Data {
public:
    Data(std::string file_name, int batch_size) :
        _batch_size(batch_size),
        _total_length(0) {
        _file.open(file_name);
        CHECK(_file.is_open()) << "file open failed";
        _file.seekg(_file.end);
        _total_length = _file.tellg();
        _file.seekg(_file.beg);
    }
    void get_batch_data(std::vector<std::vector<float>>& fea,
                        std::vector<std::vector<float>>& week_fea,
                        std::vector<std::vector<float>>& time_fea,
                        std::vector<int>& seq_offset);
private:
    std::fstream _file;
    int _total_length;
    int _batch_size;
};

void Data::get_batch_data(std::vector<std::vector<float>>& fea,
                          std::vector<std::vector<float>>& week_fea,
                          std::vector<std::vector<float>>& time_fea,
                          std::vector<int>& seq_offset) {
    CHECK(_file.is_open()) << "file open failed";
    int seq_num = 0;
    int cum = 0;

    char buf[10000];
    seq_offset.clear();
    seq_offset.push_back(0);
    fea.clear();
    week_fea.clear();
    time_fea.clear();

    while (_file.getline(buf, 10000)) {
        std::string s = buf;
        std::vector<std::string> deli_vec = {":"};
        std::vector<std::string> data_vec = string_split(s, deli_vec);

        std::vector<std::string> seq;
        seq = string_split(data_vec[0], {"|"});

        for (auto link : seq) {
            std::vector<std::string> data = string_split(link, ",");
            std::vector<float> vec;

            for (int i = 0; i < data.size(); i++) {
                vec.push_back(atof(data[i].c_str()));
            }

            fea.push_back(vec);
        }

        std::vector<std::string> week_data;
        std::vector<std::string> time_data;

        week_data = string_split(data_vec[2], ",");
        std::vector<float> vec_w;

        for (int i = 0; i < week_data.size(); i++) {
            vec_w.push_back(atof(week_data[i].c_str()));
        }

        week_fea.push_back(vec_w);

        time_data = string_split(data_vec[1], ",");
        std::vector<float> vec_t;

        for (int i = 0; i < time_data.size(); i++) {
            vec_t.push_back(atof(time_data[i].c_str()));
        }

        time_fea.push_back(vec_t);

        cum += seq.size();
        seq_offset.push_back(cum);

        seq_num++;

        if (seq_num >= _batch_size) {
            break;
        }

    }
}

void getModels(std::string path, std::vector<std::string>& files) {
    DIR* dir;
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
            getModels(path + "/" + ptr->d_name, files);
        }
    }

    closedir(dir);
}
void one_thread_run(std::string path, int thread_id) {
#ifdef USE_OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
#ifdef USE_X86_PLACE
    mkl_set_dynamic(1);
    mkl_set_num_threads(1);
#endif

    LOG(WARNING) << "load anakin model file from " << path << " ...";
    Graph<Target, Precision::FP32> graph;
    auto status = graph.load(path);

    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    LOG(INFO) << "set biggest batchsize to " << FLAGS_num;
    graph.ResetBatchSize("input_0", FLAGS_num);
    graph.ResetBatchSize("input_4", FLAGS_num);
    graph.ResetBatchSize("input_5", FLAGS_num);
    //graph.RegistOut("patch_0_stage1_unit2_conv1",  "patch_0_stage1_unit2_bn2");
    LOG(INFO) << "optimize the graph";
    graph.Optimize();
    // constructs the executer net
    LOG(INFO) << "create net to execute";
    Net<Target, Precision::FP32> net_executer(graph, true);
    // get in


    auto h_tensor_in_0 = net_executer.get_in("input_0");
    auto h_tensor_in_1 = net_executer.get_in("input_4");
    auto h_tensor_in_2 = net_executer.get_in("input_5");

    bool is_rand = false;
    Data  map_data(FLAGS_data_file, FLAGS_batch_size);
    Context<Target> ctx(0, 0, 0);
    saber::SaberTimer<Target> predict_time;


    std::vector<std::vector<std::vector<float>>> fea_vec;
    std::vector<std::vector<std::vector<float>>> week_fea_vec;
    std::vector<std::vector<std::vector<float>>> time_fea_vec;
    std::vector<std::vector<int>> seq_offset_vec;
    saber::SaberTimer<Target> load_time;
    load_time.start(ctx);
    int temp_conter = 0;

    while (true) {
        std::vector<int>* seq_offset = new std::vector<int>();
        std::vector<std::vector<float>>* fea = new std::vector<std::vector<float>>();
        std::vector<std::vector<float>>* week_fea = new std::vector<std::vector<float>>();
        std::vector<std::vector<float>>* time_fea = new std::vector<std::vector<float>>();
        map_data.get_batch_data(*fea, *week_fea, *time_fea, *seq_offset);

        if (seq_offset->size() <= 1) {
            break;
        }

        fea_vec.push_back(*fea);
        week_fea_vec.push_back(*week_fea);
        time_fea_vec.push_back(*time_fea);
        seq_offset_vec.push_back(*seq_offset);
        temp_conter++;
        //        LOG(INFO)<<seq_offset->size()<<","<<fea->size()<<","<<fea_vec.size();
    }

    load_time.end(ctx);
    float load_time_ms = load_time.get_average_ms();
    LOG(INFO) << "load time = " << load_time_ms << ",for " << temp_conter << " line data";


    predict_time.start(ctx);
    int batch_id = 0;

    for (batch_id = 0; batch_id < seq_offset_vec.size(); batch_id++) {
        std::vector<int>& seq_offset = seq_offset_vec[batch_id];
        std::vector<std::vector<float>>& fea = fea_vec[batch_id];
        std::vector<std::vector<float>>& week_fea = week_fea_vec[batch_id];
        std::vector<std::vector<float>>& time_fea = time_fea_vec[batch_id];

        //        LOG(INFO)<<seq_offset.size()<<","<<fea.size()<<","<<fea_vec.size();

        h_tensor_in_0->reshape(Shape({fea.size(), 38, 1, 1}));
        h_tensor_in_1->reshape(Shape({week_fea.size(), 10, 1, 1}));
        h_tensor_in_2->reshape(Shape({time_fea.size(), 10, 1, 1}));
        h_tensor_in_0->set_seq_offset({seq_offset});
#if defined(USE_CUDA ) || defined(AMD_GPU)
        Tensor4d<Target_H> h_tensor_0;
        Tensor4d<Target_H> h_tensor_1;
        Tensor4d<Target_H> h_tensor_2;
        h_tensor_0.reshape(h_tensor_in_0->valid_shape());
        h_tensor_1.reshape(h_tensor_in_1->valid_shape());
        h_tensor_2.reshape(h_tensor_in_2->valid_shape());

        for (int i = 0; i < fea.size(); i++) {
            memcpy(static_cast<float*>(h_tensor_0.mutable_data()) + i * 38, &fea[i][0], sizeof(float) * 38);
        }

        for (int i = 0; i < week_fea.size(); i++) {
            memcpy(static_cast<float*>(h_tensor_1.mutable_data()) + i * 10, &week_fea[i][0],
                   sizeof(float) * 10);
        }

        for (int i = 0; i < time_fea.size(); i++) {
            memcpy(static_cast<float*>(h_tensor_2.mutable_data()) + i * 10, &time_fea[i][0],
                   sizeof(float) * 10);
        }

        h_tensor_in_0->copy_from(h_tensor_0);
        h_tensor_in_1->copy_from(h_tensor_1);
        h_tensor_in_2->copy_from(h_tensor_2);
#else

        for (int i = 0; i < fea.size(); i++) {
            memcpy(static_cast<float*>(h_tensor_in_0->mutable_data()) + i * 38, &fea[i][0], sizeof(float) * 38);
        }

        for (int i = 0; i < week_fea.size(); i++) {
            memcpy(static_cast<float*>(h_tensor_in_1->mutable_data()) + i * 10, &week_fea[i][0],
                   sizeof(float) * 10);
        }

        for (int i = 0; i < time_fea.size(); i++) {
            memcpy(static_cast<float*>(h_tensor_in_2->mutable_data()) + i * 10, &time_fea[i][0],
                   sizeof(float) * 10);
        }

#endif

        net_executer.prediction();
#if defined(USE_CUDA) || defined(AMD_GPU)
#ifdef AMD_GPU
        clFinish(ctx.get_compute_stream());
#else
        cudaDeviceSynchronize();
#endif
        auto dev_out=net_executer.get_out("final_output.tmp_1_gout");
        Tensor<Target_H> out(dev_out->valid_shape());
        out.copy_from(*dev_out);
        int size = out.valid_size();

        for (int seq_id = 0; seq_id < seq_offset.size() - 1; seq_id++) {
            int seq_len = seq_offset[seq_id + 1] - seq_offset[seq_id];
            int seq_start = seq_offset[seq_id];

            for (int i = 0; i < seq_len - 1; i++) {
                printf("%f|", static_cast<float*>(out.data())[seq_start + i]);
            }

            printf("%f\n", static_cast<float*>(out.data())[seq_start + seq_len - 1]);
        }
#else
        auto out =net_executer.get_out("final_output.tmp_1_gout");
        int size = out->valid_size();

        for (int seq_id = 0; seq_id < seq_offset.size() - 1; seq_id++) {
            int seq_len = seq_offset[seq_id + 1] - seq_offset[seq_id];
            int seq_start = seq_offset[seq_id];

            for (int i = 0; i < seq_len - 1; i++) {
                printf("%f|", static_cast<float*>(out->data())[seq_start + i]);
            }

            printf("%f\n", static_cast<float*>(out->data())[seq_start + seq_len - 1]);
        }
#endif



        //break;
    }

    predict_time.end(ctx);


    float time_ms = predict_time.get_average_ms();
    LOG(INFO) << "[result]: thread_id = " << thread_id << "," << " batch_size " << FLAGS_batch_size
              << " avg time " << time_ms / batch_id << " ms" << ", total time = " << time_ms;

}

TEST(NetTest, net_execute_base_test) {
    std::vector<std::string> models;
    Env<Target>::env_init();
    Context<Target> ctx(0, 0, 1);
    SaberTimer<Target> timer;
    timer.start(ctx);
    std::vector<std::unique_ptr<std::thread>> threads;

    for (int thread_id = 0; thread_id < FLAGS_thread_num; thread_id++) {
        threads.emplace_back(new std::thread(&one_thread_run, FLAGS_model_file, thread_id));
    }

    for (int i = 0; i < FLAGS_thread_num; ++i) {
        threads[i]->join();
    }

    timer.end(ctx);
    float time_consume = timer.get_average_ms();
    LOG(INFO) << "[result]: totol time = " << time_consume << " ms, QPS = " << FLAGS_num*
              FLAGS_thread_num / time_consume * 1000
              << " , thread num = " << FLAGS_thread_num;
    LOG(WARNING) << "load anakin model file from " << FLAGS_model_file << " ...";


}

int main(int argc, const char** argv) {

    Env<Target>::env_init();

    // initial logger
    logger::init(argv[0]);


    LOG(INFO) << "BenchMark usage:";
    LOG(INFO) << "   $benchmark <model_dir> <model_file> <num> <warmup_iter> <epoch>";
    LOG(INFO) << "   model_dir:      model directory";
    LOG(INFO) << "   model_file:     path to model";
    LOG(INFO) << "   num:            batchSize default to 1";
    LOG(INFO) << "   batchsize:      batchsize to 1";
    LOG(INFO) << "   thread_number:  thread_number default to 1";


    if (argc < 3) {
        LOG(ERROR) << "You should fill in the variable model_dir and model_file at least.";
        return 0;
    }

    FLAGS_data_file = argv[1];

    if (argc > 2) {
        FLAGS_model_file = argv[2];
    }

    if (argc > 3) {
        FLAGS_num = atoi(argv[3]);
    }

    if (argc > 4) {
        FLAGS_batch_size = atoi(argv[4]);
    }

    if (argc > 5) {
        FLAGS_thread_num = atoi(argv[5]);
    }


    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

