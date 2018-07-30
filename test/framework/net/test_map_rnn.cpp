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
int FLAGS_batch_size = 1;
#endif

std::vector<std::string> string_split(std::string in_str, std::string delimiter) {
    std::vector<std::string> seq;
    int found = in_str.find(delimiter);
    int pre_found = -1;
    while (found != std::string::npos) {
        if (pre_found == -1) {
            seq.push_back(in_str.substr(0, found));
        } else {
            seq.push_back(in_str.substr(pre_found + delimiter.length(), found - delimiter.length() - pre_found));
        }
        pre_found = found;
        found = in_str.find(delimiter, pre_found + delimiter.length());
    }
    seq.push_back(in_str.substr(pre_found+1, in_str.length() - (pre_found+1)));
    return seq;
}
std::vector<std::string> string_split(std::string in_str, std::vector<std::string> &delimiter) {
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

class Data{
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
    while(_file.getline(buf, 10000)) {
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
        std::string data_path = "/home/chengyujuan/test_features_sys";
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
           while(true) {
               LOG(INFO)<<batch_id++;
               seq_offset.clear();
               map_data.get_batch_data(fea, week_fea, time_fea, seq_offset);
               if (seq_offset.size() <= 1) {
                   break;
               }
               h_tensor_in_0->reshape(Shape(fea.size(), 38, 1, 1));
               h_tensor_in_1->reshape(Shape(week_fea.size(), 10, 1, 1));
               h_tensor_in_2->reshape(Shape(time_fea.size(), 10, 1, 1));
               h_tensor_in_0->set_seq_offset(seq_offset);
               for (int i = 0; i < fea.size(); i++) {
                   memcpy(h_tensor_in_0->mutable_data() + i * 38, &fea[i][0], sizeof(float)*38);
               }
               for (int i = 0; i < week_fea.size(); i++) {
                   memcpy(h_tensor_in_1->mutable_data() + i * 10, &week_fea[i][0], sizeof(float)*10);
               }
               for (int i = 0; i < time_fea.size(); i++) {
                   memcpy(h_tensor_in_2->mutable_data() + i * 10, &time_fea[i][0], sizeof(float)*10);
               }
                net_executer.prediction();
                //cudaDeviceSynchronize();
                auto out = net_executer.get_out("final_output.tmp_1_gout");
                //print_tensor_host(*out);
                break;
           }
           my_time.end(ctx);
       }

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
        std::string save_model_path = *iter + std::string(".saved");
        status = graph.save(save_model_path);
        if (!status ) {
            LOG(FATAL) << " [ERROR] " << status.info();
        }
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
    LOG(INFO)<< "   batch_size:          time statistic epoch default to 1000";
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
        FLAGS_batch_size = atoi(argv[6]);
    }
#endif
    InitTest();
    RUN_ALL_TESTS(argv[0]); 
    return 0;
}
