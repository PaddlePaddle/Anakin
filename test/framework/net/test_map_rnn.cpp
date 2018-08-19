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
#include <mkl_service.h>

#if defined(USE_CUDA)
using Target = NV;
using Target_H = NVHX86;
#elif defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#elif defined(USE_ARM_PLACE)
using Target = ARM;
using Target_H = ARM;
#endif

#ifdef USE_GFLAGS
#include <gflags/gflags.h>

DEFINE_string(model_dir, "", "model dir");
DEFINE_string(model_file, "", "model file");
DEFINE_int32(num, 1, "batchSize");
DEFINE_int32(warmup_iter, 10, "warm up iterations");
DEFINE_int32(epoch, 1000, "time statistic epoch");
DEFINE_int32(batch_size, 1, "seq num");
DEFINE_int32(thread_num, 1, "thread_num");
#else
std::string FLAGS_model_dir;
std::string FLAGS_model_file;
int FLAGS_num = 1;
int FLAGS_warmup_iter = 10;
int FLAGS_epoch = 1000;
int FLAGS_batch_size = 1;
int FLAGS_thread_num = 1;
#endif

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
void one_thread_run(std::string path,int thread_id){
#ifdef USE_OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
#ifdef USE_X86_PLACE
    mkl_set_num_threads(1);
#endif

            LOG(WARNING) << "load anakin model file from " << path << " ...";
    Graph<Target, AK_FLOAT, Precision::FP32> graph;
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
    Net<Target, AK_FLOAT, Precision::FP32> net_executer(graph, true);
    // get in


    auto h_tensor_in_0 = net_executer.get_in("input_0");
    auto h_tensor_in_1 = net_executer.get_in("input_4");
    auto h_tensor_in_2 = net_executer.get_in("input_5");

    bool is_rand = false;
    std::string data_path = "./test_features_sys";
    Data  map_data(data_path, FLAGS_batch_size);
    Context<Target> ctx(0, 0, 0);
    saber::SaberTimer<Target> my_time;


    std::vector<std::vector<float>> fea;
    std::vector<std::vector<float>> week_fea;
    std::vector<std::vector<float>> time_fea;
    std::vector<int> seq_offset;
    my_time.start(ctx);
    int batch_id = 0;

    while (true) {
        seq_offset.clear();
        map_data.get_batch_data(fea, week_fea, time_fea, seq_offset);

        if (seq_offset.size() <= 1) {
            break;
        }

        h_tensor_in_0->reshape(Shape(fea.size(), 38, 1, 1));
        h_tensor_in_1->reshape(Shape(week_fea.size(), 10, 1, 1));
        h_tensor_in_2->reshape(Shape(time_fea.size(), 10, 1, 1));
        h_tensor_in_0->set_seq_offset(seq_offset);
#ifdef USE_CUDA
        Tensor4d<Target_H, AK_FLOAT> h_tensor_0;
        Tensor4d<Target_H, AK_FLOAT> h_tensor_1;
        Tensor4d<Target_H, AK_FLOAT> h_tensor_2;
        h_tensor_0.reshape(h_tensor_in_0->valid_shape());
        h_tensor_1.reshape(h_tensor_in_1->valid_shape());
        h_tensor_2.reshape(h_tensor_in_2->valid_shape());

        for (int i = 0; i < fea.size(); i++) {
            memcpy(h_tensor_0.mutable_data() + i * 38, &fea[i][0], sizeof(float) * 38);
        }

        for (int i = 0; i < week_fea.size(); i++) {
            memcpy(h_tensor_1.mutable_data() + i * 10, &week_fea[i][0], sizeof(float) * 10);
        }

        for (int i = 0; i < time_fea.size(); i++) {
            memcpy(h_tensor_2.mutable_data() + i * 10, &time_fea[i][0], sizeof(float) * 10);
        }

        h_tensor_in_0->copy_from(h_tensor_0);
        h_tensor_in_1->copy_from(h_tensor_1);
        h_tensor_in_2->copy_from(h_tensor_2);
#else

        for (int i = 0; i < fea.size(); i++) {
                memcpy(h_tensor_in_0->mutable_data() + i * 38, &fea[i][0], sizeof(float) * 38);
            }

            for (int i = 0; i < week_fea.size(); i++) {
                memcpy(h_tensor_in_1->mutable_data() + i * 10, &week_fea[i][0], sizeof(float) * 10);
            }

            for (int i = 0; i < time_fea.size(); i++) {
                memcpy(h_tensor_in_2->mutable_data() + i * 10, &time_fea[i][0], sizeof(float) * 10);
            }

#endif

        net_executer.prediction();
        batch_id++;
#ifdef USE_CUDA
        cudaDeviceSynchronize();
        auto out = net_executer.get_out("final_output.tmp_1_gout");
        //print_tensor_device(*out);

        cudaDeviceSynchronize();
#endif
//        auto out = net_executer.get_out("final_output.tmp_1_gout");
//        int size=out->valid_size();
//        for(int i=0;i<size-1;i++){
//            printf("%f|",out->data()[i] );
//        }
//        printf("%f\n", out->data()[size-1]);
        //break;
    }

    my_time.end(ctx);


    size_t end = (path).find(".anakin.bin");
    size_t start = FLAGS_model_dir.length();
    std::string model_name = (path).substr(start, end - start);
    float time_ms=my_time.get_average_ms();
            LOG(INFO)<<"[result]: thread_id = "<<thread_id<<"," << model_name << " batch_size " << FLAGS_batch_size
                     << " avg time " <<time_ms/ batch_id << " ms"<<", total time = "<<time_ms;

}

TEST(NetTest, net_execute_base_test) {
    std::vector<std::string> models;
    Env<X86>::env_init();

    if (FLAGS_model_file == "") {
        getModels(FLAGS_model_dir, models);
    } else {
        models.push_back(FLAGS_model_dir + FLAGS_model_file);
    }

    for (auto iter = models.begin(); iter < models.end(); iter++) {
        Context<X86> ctx(0,0,1);
        SaberTimer<X86> timer;
        timer.start(ctx);
        std::vector<std::unique_ptr<std::thread>> threads;
        for(int thread_id=0;thread_id<FLAGS_thread_num;thread_id++){
            threads.emplace_back(new std::thread(&one_thread_run,*iter,thread_id));
        }
        for (int i = 0; i < FLAGS_thread_num; ++i) {
            threads[i]->join();
        }
        timer.end(ctx);
        float time_consume=timer.get_average_ms();
                LOG(INFO) <<"[result]: totol time = "<<time_consume<<" ms, QPS = "<<FLAGS_num*FLAGS_thread_num/time_consume*1000
                          <<" , thread num = "<<FLAGS_thread_num;
                LOG(WARNING) << "load anakin model file from " << *iter << " ...";
    }

}




#define  RUN_IN_WORKER
#ifdef RUN_IN_WORKER
//void consumer_task(Worker<Target , AK_FLOAT, Precision::FP32> *workers){
//    LOG(INFO)<<"hello_world";
//    int iterator = 88656;
//    int count=0;
//    while(iterator) {
//        if(!workers->empty()) {
//            LOG(INFO)<<"hello_world2";
//            auto d_tensor_p = workers->async_get_result();
//            LOG(INFO)<<"consumer "<<count;
//            count++;
//            Tensor<Target, AK_FLOAT>* out=d_tensor_p[0];
//            iterator--;
//        }
//    }
//    LOG(INFO)<<"bye_world";
//}
//TEST(NetTest, net_execute_base_test_worker) {
//    std::vector<std::string> models;
//
//    if (FLAGS_model_file == "") {
//        getModels(FLAGS_model_dir, models);
//    } else {
//        models.push_back(FLAGS_model_dir + FLAGS_model_file);
//    }
//
//    for (auto iter = models.begin(); iter < models.end(); iter++) {
//        Worker<Target , AK_FLOAT, Precision::FP32>  workers(*iter, 1);
//        Graph<Target, AK_FLOAT, Precision::FP32> graph;
//        auto status = graph.load(*iter);
//        if (!status) {
//            LOG(FATAL) << " [ERROR] " << status.info();
//        }
//        std::vector<std::string>& vout_name = graph.get_outs();
//        std::vector<std::string> input_names={"input_0","input_4","input_5"};
//        workers.register_inputs(input_names);
//        workers.register_outputs(vout_name);
//        /*.Reshape("input_0", {FLAGS_num, 38, 1, 1});
//        workers.Reshape("input_4", {FLAGS_num, 10, 1, 1});
//        workers.Reshape("input_5", {FLAGS_num, 10, 1, 1});*/
//        workers.launch();
//
//        auto h_tensor_in_0 = new Tensor<Target_H,AK_FLOAT>(Shape(FLAGS_num, 38, 1, 1));
//        auto h_tensor_in_1 = new Tensor<Target_H,AK_FLOAT>(Shape(FLAGS_num, 10, 1, 1));
//        auto h_tensor_in_2 = new Tensor<Target_H,AK_FLOAT>(Shape(FLAGS_num, 10, 1, 1));
//
//        LOG(INFO)<<"init batchsize = "<<FLAGS_num;
//        std::vector<Tensor<Target_H, AK_FLOAT>*> inputs={h_tensor_in_0,h_tensor_in_1,h_tensor_in_2};
//
//        std::string data_path = "./test_features_sys";
//        Data  map_data(data_path, FLAGS_batch_size);
//        Context<Target> ctx(0, 0, 0);
//        saber::SaberTimer<Target> my_time;
//        std::vector<std::vector<float>> fea;
//        std::vector<std::vector<float>> week_fea;
//        std::vector<std::vector<float>> time_fea;
//        std::vector<int> seq_offset;
//        my_time.start(ctx);
//        int batch_id = 0;
//        std::thread* consumer=new std::thread(&consumer_task,&workers);
//        while (true) {
//            LOG(INFO) << batch_id++;
//            seq_offset.clear();
//            map_data.get_batch_data(fea, week_fea, time_fea, seq_offset);
//
//            if (seq_offset.size() <= 1) {
//                break;
//            }
//
//            /*h_tensor_in_0->reshape(Shape(fea.size(), 38, 1, 1));
//            h_tensor_in_1->reshape(Shape(week_fea.size(), 10, 1, 1));
//            h_tensor_in_2->reshape(Shape(time_fea.size(), 10, 1, 1));
//            h_tensor_in_0->set_seq_offset(seq_offset);
//
//            for (int i = 0; i < fea.size(); i++) {
//                memcpy(h_tensor_in_0->mutable_data() + i * 38, &fea[i][0], sizeof(float) * 38);
//            }
//
//            for (int i = 0; i < week_fea.size(); i++) {
//                memcpy(h_tensor_in_1->mutable_data() + i * 10, &week_fea[i][0], sizeof(float) * 10);
//            }
//
//            for (int i = 0; i < time_fea.size(); i++) {
//                memcpy(h_tensor_in_2->mutable_data() + i * 10, &time_fea[i][0], sizeof(float) * 10);
//            }*/
//
////            workers.sync_prediction(inputs);
//            workers.async_prediction(inputs);
//
//        }
//        consumer->join();
//        LOG(INFO) << "batch_id = " <<batch_id;
//
//
//
//        my_time.end(ctx);
//
//        size_t end = (*iter).find(".anakin.bin");
//        size_t start = FLAGS_model_dir.length();
//        std::string model_name = (*iter).substr(start, end - start);
//
//        LOG(INFO) << model_name << " batch_size " << FLAGS_num << " average time " <<
//                  my_time.get_average_ms() / FLAGS_epoch << " ms";
//        std::string save_model_path = *iter + std::string(".saved");
//        status = graph.save(save_model_path);
//
//        if (!status) {
//            LOG(FATAL) << " [ERROR] " << status.info();
//        }
//    }
//}

#else
TEST(NetTest, net_execute_base_test) {
    std::vector<std::string> models;

    if (FLAGS_model_file == "") {
        getModels(FLAGS_model_dir, models);
    } else {
        models.push_back(FLAGS_model_dir + FLAGS_model_file);
    }

    for (auto iter = models.begin(); iter < models.end(); iter++) {
        LOG(WARNING) << "load anakin model file from " << *iter << " ...";
        Graph<Target, AK_FLOAT, Precision::FP32> graph;
        auto status = graph.load(*iter);

        if (!status) {
            LOG(FATAL) << " [ERROR] " << status.info();
        }

        LOG(INFO) << "set batchsize to " << FLAGS_num;
        graph.ResetBatchSize("input_0", FLAGS_num);
        graph.ResetBatchSize("input_4", FLAGS_num);
        graph.ResetBatchSize("input_5", FLAGS_num);
        //graph.RegistOut("patch_0_stage1_unit2_conv1",  "patch_0_stage1_unit2_bn2");
        LOG(INFO) << "optimize the graph";
        graph.Optimize();
        // constructs the executer net
        LOG(INFO) << "create net to execute";
        Net<Target, AK_FLOAT, Precision::FP32> net_executer(graph, true);
        // get in
        LOG(INFO) << "get input";

        auto h_tensor_in_0 = net_executer.get_in("input_0");
        auto h_tensor_in_1 = net_executer.get_in("input_4");
        auto h_tensor_in_2 = net_executer.get_in("input_5");
        //fill_tensor_host_rand(*h_tensor_in_0, -1.f, 1.f);
        //fill_tensor_host_rand(*h_tensor_in_1, -1.f, 1.f);
        //fill_tensor_host_rand(*h_tensor_in_2, -1.f, 1.f);
        bool is_rand = false;
        std::string data_path = "./test_features_sys";
        Data  map_data(data_path, FLAGS_batch_size);
        Context<Target> ctx(0, 0, 0);
        saber::SaberTimer<Target> my_time;

        if (is_rand) {
            int cum = 0;
            std::vector<int> seq_offset;
            seq_offset.push_back(cum);

            for (int i = 0; i < FLAGS_batch_size; i++) {
                //int len = std::rand() % 60 + 1;
                int len = 30;
                cum += len;

                seq_offset.push_back(cum);
            }

            h_tensor_in_0->reshape(Shape(cum, 38, 1, 1));
            h_tensor_in_0->set_seq_offset(seq_offset);

            Tensor4d<Target_H, AK_FLOAT> h_tensor_0;
            Tensor4d<Target_H, AK_FLOAT> h_tensor_1;
            Tensor4d<Target_H, AK_FLOAT> h_tensor_2;
            h_tensor_0.reshape(h_tensor_in_0->valid_shape());
            h_tensor_1.reshape(h_tensor_in_1->valid_shape());
            h_tensor_2.reshape(h_tensor_in_2->valid_shape());
            fill_tensor_host_rand(h_tensor_0);
            fill_tensor_host_rand(h_tensor_1);
            fill_tensor_host_rand(h_tensor_2);
            h_tensor_in_0->copy_from(h_tensor_0);
            h_tensor_in_1->copy_from(h_tensor_1);
            h_tensor_in_2->copy_from(h_tensor_2);
            LOG(WARNING) << "EXECUTER !!!!!!!! ";
#ifdef USE_CUDA
            cudaDeviceSynchronize();
#endif

            for (int i = 0; i < FLAGS_warmup_iter; i++) {
                net_executer.prediction();
                //cudaDeviceSynchronize();
                //auto out = net_executer.get_out("patch_0_pre_fc1");
                //print_tensor_device(*out);
            }

#ifdef ENABLE_OP_TIMER
            net_executer.reset_op_time();
#endif
            my_time.start(ctx);

            //auto start = std::chrono::system_clock::now();
            for (int i = 0; i < FLAGS_epoch; i++) {
                //DLOG(ERROR) << " epoch(" << i << "/" << epoch << ") ";
                net_executer.prediction();
            }

            my_time.end(ctx);
        } else {
            std::vector<std::vector<float>> fea;
            std::vector<std::vector<float>> week_fea;
            std::vector<std::vector<float>> time_fea;
            std::vector<int> seq_offset;
            my_time.start(ctx);
            int batch_id = 0;

            while (true) {
                LOG(INFO) << batch_id++;
                seq_offset.clear();
                map_data.get_batch_data(fea, week_fea, time_fea, seq_offset);

                if (seq_offset.size() <= 1) {
                    break;
                }

                h_tensor_in_0->reshape(Shape(fea.size(), 38, 1, 1));
                h_tensor_in_1->reshape(Shape(week_fea.size(), 10, 1, 1));
                h_tensor_in_2->reshape(Shape(time_fea.size(), 10, 1, 1));
                h_tensor_in_0->set_seq_offset(seq_offset);
#ifdef USE_CUDA
                Tensor4d<Target_H, AK_FLOAT> h_tensor_0;
                Tensor4d<Target_H, AK_FLOAT> h_tensor_1;
                Tensor4d<Target_H, AK_FLOAT> h_tensor_2;
                h_tensor_0.reshape(h_tensor_in_0->valid_shape());
                h_tensor_1.reshape(h_tensor_in_1->valid_shape());
                h_tensor_2.reshape(h_tensor_in_2->valid_shape());

                for (int i = 0; i < fea.size(); i++) {
                    memcpy(h_tensor_0.mutable_data() + i * 38, &fea[i][0], sizeof(float) * 38);
                }

                for (int i = 0; i < week_fea.size(); i++) {
                    memcpy(h_tensor_1.mutable_data() + i * 10, &week_fea[i][0], sizeof(float) * 10);
                }

                for (int i = 0; i < time_fea.size(); i++) {
                    memcpy(h_tensor_2.mutable_data() + i * 10, &time_fea[i][0], sizeof(float) * 10);
                }

                h_tensor_in_0->copy_from(h_tensor_0);
                h_tensor_in_1->copy_from(h_tensor_1);
                h_tensor_in_2->copy_from(h_tensor_2);
#else

                for (int i = 0; i < fea.size(); i++) {
                    memcpy(h_tensor_in_0->mutable_data() + i * 38, &fea[i][0], sizeof(float) * 38);
                }

                for (int i = 0; i < week_fea.size(); i++) {
                    memcpy(h_tensor_in_1->mutable_data() + i * 10, &week_fea[i][0], sizeof(float) * 10);
                }

                for (int i = 0; i < time_fea.size(); i++) {
                    memcpy(h_tensor_in_2->mutable_data() + i * 10, &time_fea[i][0], sizeof(float) * 10);
                }

#endif

                net_executer.prediction();
#ifdef USE_CUDA
                cudaDeviceSynchronize();
                auto out = net_executer.get_out("final_output.tmp_1_gout");
                //print_tensor_device(*out);

                cudaDeviceSynchronize();
#endif
                //break;
            }

            my_time.end(ctx);
        }

#ifdef ENABLE_OP_TIMER
        std::vector<float> op_time = net_executer.get_op_time();
        auto exec_funcs = net_executer.get_exec_funcs();
        auto op_param = net_executer.get_op_param();

        for (int i = 0; i <  op_time.size(); i++) {
            LOG(INFO) << "name: " << exec_funcs[i].name << " op_type: " << exec_funcs[i].op_name <<
                      " op_param: " << op_param[i] << " time " << op_time[i] / FLAGS_epoch;
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
            LOG(INFO) << it->first << "  " << (it->second) / FLAGS_epoch << " ms";
        }

#endif
        size_t end = (*iter).find(".anakin.bin");
        size_t start = FLAGS_model_dir.length();
        std::string model_name = (*iter).substr(start, end - start);

        LOG(INFO) << model_name << " batch_size " << FLAGS_num << " average time " <<
                  my_time.get_average_ms() / FLAGS_epoch << " ms";
        std::string save_model_path = *iter + std::string(".saved");
        status = graph.save(save_model_path);

        if (!status) {
            LOG(FATAL) << " [ERROR] " << status.info();
        }
    }
}

#endif
int main(int argc, const char** argv) {

    Env<Target>::env_init();

    // initial logger
    logger::init(argv[0]);

#ifdef USE_GFLAGS
    google::ParseCommandLineFlags(&argc, &argv, true);
#else
            LOG(INFO) << "BenchMark usage:";
            LOG(INFO) << "   $benchmark <model_dir> <model_file> <num> <warmup_iter> <epoch>";
            LOG(INFO) << "   model_dir:      model directory";
            LOG(INFO) << "   model_file:     path to model";
            LOG(INFO) << "   num:            batchSize default to 1";
            LOG(INFO) << "   warmup_iter:    warm up iterations default to 10";
            LOG(INFO) << "   epoch:          time statistic epoch default to 1000";
            LOG(INFO) << "   batch_size:          time statistic epoch default to 1000";

    if (argc < 3) {
                LOG(ERROR) << "You should fill in the variable model_dir and model_file at least.";
        return 0;
    }

    FLAGS_model_dir = argv[1];

    if (argc > 2) {
        FLAGS_model_file = argv[2];
    }

    if (argc > 3) {
        FLAGS_num = atoi(argv[3]);
    }

    if (argc > 4) {
        FLAGS_warmup_iter = atoi(argv[4]);
    }

    if (argc > 5) {
        FLAGS_epoch = atoi(argv[5]);
    }

    if (argc > 6) {
        FLAGS_batch_size = atoi(argv[6]);
    }
    if (argc > 7) {
        FLAGS_thread_num = atoi(argv[7]);
    }

#endif
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
