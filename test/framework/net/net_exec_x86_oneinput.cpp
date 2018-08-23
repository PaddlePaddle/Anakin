
#include "anakin_config.h"
#include <string>
#include <fstream>
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

#include "sys/time.h"
#include "debug.h"
#include "string"

#ifdef USE_X86_PLACE
#include <mkl_service.h>
#include "omp.h"
#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(int, run_threads, 1);
volatile DEFINE_GLOBAL(int, batch_size, 1);
volatile DEFINE_GLOBAL(int, max_word_len, 0);
volatile DEFINE_GLOBAL(int, word_count, 0);
DEFINE_GLOBAL(std::string, model_dir, "");
DEFINE_GLOBAL(std::string, input_file, "");
DEFINE_GLOBAL(std::string, split_word, "\t");
DEFINE_GLOBAL(std::string, output_name, "");
DEFINE_GLOBAL(std::string, run_mode, "instance");
DEFINE_GLOBAL(int, split_index, 0);

#define AVG_INPUT
void getModels(std::string path, std::vector<std::string>& files) {
    DIR* dir = nullptr;
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

int read_file(std::vector<float>& results, const char* file_name) {

    std::ifstream infile(file_name);

    if (!infile.good()) {
        std::cout << "Cannot open " << std::endl;
        return false;
    }

    LOG(INFO) << "found filename: " << file_name;
    std::string line;

    while (std::getline(infile, line)) {
        results.push_back((float)atof(line.c_str()));
    }

    return 0;
}
void SplitString(const std::string& s,
                 std::vector<std::string>& v, const std::string& c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }

    if (pos1 != s.length()) {
        v.push_back(s.substr(pos1));
    }
}

int split_word_from_file(
    std::vector<std::vector<float> >& word_idx,
    const std::string input_file_path,
    const std::string split_token,
    const std::string inner_split_token,
    const int col_select) {

    std::ifstream infile(input_file_path.c_str());

    if (!infile.good()) {
        std::cout << "Cannot open " << std::endl;
        return 1;
    }

    LOG(INFO) << "found filename: " << input_file_path;
    std::string line;
    std::vector<std::string> split_v;
    std::vector<std::string> split_w;
    int word_count = 0;

    while (std::getline(infile, line)) {
        split_v.clear();
        SplitString(line, split_v, split_token);
        CHECK_GE(split_v.size(), col_select + 1) << " file need ; split";
        std::vector<float> word;
        std::vector<float> mention;
        split_w.clear();
        SplitString(split_v[col_select], split_w, inner_split_token);

        for (auto w : split_w) {
            word.push_back(atof(w.c_str()));
            word_count++;
            //            printf("%d,",atoi(w.c_str()));
        }

        //        printf("\n");
        //        exit(0);
        word_idx.push_back(word);
    }

    GLB_word_count = word_count;
    return 0;
}

int get_batch_data_offset(
    std::vector<float>& out_data,
    const std::vector<std::vector<float> >& seq_data,
    std::vector<int>& seq_offset,
    const int start_idx,
    const int batch_num) {

    seq_offset.clear();
    out_data.clear();
    seq_offset.push_back(0);
    int len = 0;

    for (int i = 0; i < batch_num; ++i) {
        for (auto d : seq_data[i + start_idx]) {
            len += 1;
            out_data.push_back(d);
            //            printf("%.0f, ",d);
        }

        //        printf("\n");
        seq_offset.push_back(len);
    }

    return len;
}

void anakin_net_thread(std::vector<Tensor4dPtr<X86, AK_FLOAT> >* data_in, std::string model_path
                      , bool is_save_result=false,int thread_id=0) {
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    mkl_set_num_threads(1);
    Graph<X86, AK_FLOAT, Precision::FP32>* graph = new Graph<X86, AK_FLOAT, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << model_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    graph->Reshape("input_0", {GLB_max_word_len, 1, 1, 1});
    //anakin graph optimization
    graph->Optimize();
    Net<X86, AK_FLOAT, Precision::FP32> net_executer(*graph, true);
    //    SaberTimer<X86> timer;
    //    Context<X86> ctx;
    struct timeval time_start, time_end;

    int slice_10 = data_in->size() / 10;
    slice_10 = slice_10 > 0 ? slice_10 : 1;
    int word_sum = 0;
    gettimeofday(&time_start, nullptr);

    for (int i = 0; i < data_in->size(); ++i) {
        auto input_tensor = (*data_in)[i];
        auto word_in_p = net_executer.get_in("input_0");
        int len_sum = input_tensor->valid_size();
        word_in_p->reshape({len_sum, 1, 1, 1});
        word_sum += len_sum;

        for (int j = 0; j < len_sum; ++j) {
            word_in_p->mutable_data()[j] = input_tensor->data()[j];
        }

        word_in_p->set_seq_offset(input_tensor->get_seq_offset());
        //        timer.start(ctx);

        net_executer.prediction();
        if(is_save_result) {
            int k = net_executer.get_out_list().size();
                    LOG(INFO) << "out size = " << k;
            auto outs = net_executer.get_out(GLB_output_name);
            int out_cnt = 0;
            record_dev_tensorfile(outs,
                                  ("output_" + std::to_string(thread_id) + "_" + std::to_string(i)+ ".txt").data());
        }
//        out_cnt++;

        //        timer.end(ctx);
        if (i % slice_10 == 0) {
            LOG(INFO) << "thread run " << i << " of " << data_in->size();
        }
    }

    gettimeofday(&time_end, nullptr);
    float use_ms = (time_end.tv_sec - time_start.tv_sec) * 1000.f + (time_end.tv_usec -
                   time_start.tv_usec) / 1000.f;
    LOG(INFO) << "summary_thread :thread total : " << use_ms << " ms, avg = " <<
              (use_ms / data_in->size()) << ", consume words = " << word_sum;
    free(graph);
}


std::string get_model_path(){
    std::vector<std::string> models;
    getModels(GLB_model_dir, models);
    return models[0];
}
std::vector<std::vector<float> > get_input_data(){
    std::vector<std::vector<float> > word_idx;
    if (split_word_from_file(word_idx, GLB_input_file, GLB_split_word, " ", GLB_split_index)) {
                LOG(ERROR) << " NOT FOUND " << GLB_input_file;
        exit(-1);
    }
    return word_idx;
};
std::vector<std::vector<Tensor<X86, AK_FLOAT>* >> get_slice_input_data(std::vector<std::vector<float> > &word_idx,
                                                                       int thread_num,int &real_max_batch_word_len,int batch_num){
    std::vector<std::vector<Tensor<X86, AK_FLOAT>* >> host_tensor_p_in_list;
    std::vector<float> word_idx_data;
    std::vector<int> word_seq_offset;
    for (int tid = 0; tid < thread_num; ++tid) {
        std::vector<Tensor<X86, AK_FLOAT>* > data4thread;
        int start_wordid = tid * (word_idx.size() / thread_num);
        int end_wordid = (tid + 1) * (word_idx.size() / thread_num);

        for (int i = start_wordid; i < end_wordid; i += batch_num) {
            int word_len = get_batch_data_offset(word_idx_data, word_idx, word_seq_offset, i, batch_num);
            real_max_batch_word_len = real_max_batch_word_len < word_len ? word_len : real_max_batch_word_len;
            saber::Shape valid_shape({word_len, 1, 1, 1});
            Tensor4d<X86, AK_FLOAT>* tensor_p = new Tensor4d<X86, AK_FLOAT>(valid_shape);
                    CHECK_EQ(word_len, word_idx_data.size()) << "word_len == word_idx_data.size";

            for (int j = 0; j < word_idx_data.size(); ++j) {
                tensor_p->mutable_data()[j] = word_idx_data[j];
            }

            tensor_p->set_seq_offset(word_seq_offset);
            data4thread.push_back(tensor_p);
        }
        host_tensor_p_in_list.push_back(data4thread);
    }
    return host_tensor_p_in_list;
};
void instance_run(){
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    mkl_set_num_threads(1);

    std::string model_path=get_model_path();


    std::vector<std::vector<float> > word_idx;
    word_idx=get_input_data();

    int batch_num = GLB_batch_size;
    int thread_num = GLB_run_threads;

    int real_max_batch_word_len = 0;


    std::vector<std::vector<Tensor<X86, AK_FLOAT>* >> host_tensor_p_in_list;
#ifdef AVG_INPUT
    host_tensor_p_in_list=get_slice_input_data(word_idx,1,real_max_batch_word_len,batch_num);
#else
    host_tensor_p_in_list=get_slice_input_data(word_idx,thread_num,real_max_batch_word_len,batch_num);
#endif


    GLB_max_word_len = real_max_batch_word_len;

    if(real_max_batch_word_len<=0){
        LOG(INFO)<<"can`t read any data";
        exit(-1);
    }

    for (int i = 0; i < GLB_split_word.size(); i++) {
        LOG(INFO) << "splite  " << i << " in " << GLB_split_word.size() << "  is " << int(char(
        GLB_split_word[i]));
    }

    std::string first_line="";
    Tensor<X86, AK_FLOAT>* first_tensor=host_tensor_p_in_list[0][0];
    for(int  i=0 ;i<first_tensor->valid_size() ;++i){
        first_line=first_line+std::to_string(first_tensor->data()[i])+",";
        LOG(INFO)<<"first line : "<<first_tensor->data()[i];
    }
    LOG(INFO)<<"first line2 :"<<first_line;

    LOG(WARNING) << "Async Runing multi_threads for model: " << model_path << ",batch dim = " <<
                 batch_num
                 << ",line num = " << word_idx.size() << ", number of word = " << GLB_word_count <<
                 ",thread number size = " << thread_num << ",real max = " << real_max_batch_word_len;

    std::vector<std::unique_ptr<std::thread>> threads;
    struct timeval time_start, time_end;

    gettimeofday(&time_start, nullptr);

    for (int i = 0; i < thread_num; ++i) {
#ifdef AVG_INPUT
        threads.emplace_back(
                new std::thread(&anakin_net_thread, &host_tensor_p_in_list[0], model_path,GLB_output_name!="",i));
#else
        threads.emplace_back(
                new std::thread(&anakin_net_thread, &host_tensor_p_in_list[i], model_path,GLB_output_name!="",i));
#endif
        //        threads.emplace_back(
        //                new std::thread(&anakin_net_thread, &host_tensor_p_in_list[i]),models[0]);
    }

    for (int i = 0; i < thread_num; ++i) {
        threads[i]->join();
    }

    gettimeofday(&time_end, nullptr);
    float use_ms = (time_end.tv_sec - time_start.tv_sec) * 1000.f + (time_end.tv_usec -
                                                                     time_start.tv_usec) / 1000.f;

            LOG(INFO) << "summary: " << "thread num = " << thread_num << ",total time = " << use_ms <<
                      "ms ,batch = " << batch_num
                      << ",word sum = " << GLB_word_count << ", seconde/line = " << (use_ms / word_idx.size())
#ifdef AVG_INPUT
                      << ",AVG_INPUT QPS  = " << (thread_num*word_idx.size() / use_ms * 1000)
                      <<"line/second, "<<(GLB_word_count*thread_num)/use_ms*1000<<" words/second";
#else
                         << ",QPS = " << (word_idx.size() / use_ms * 1000);
                         <<"line/second, "<<(GLB_word_count)/use_ms*1000<<" words/second";
#endif
}
void worker_run(){
    std::string model_path=get_model_path();
    Worker<X86, AK_FLOAT, Precision::FP32>  workers(model_path, 10);
    workers.register_inputs({"input_0"});
    if(GLB_output_name!=""){
        workers.register_outputs({GLB_output_name});
    }

    std::vector<std::vector<float> > word_idx;
    word_idx=get_input_data();

    int batch_num = GLB_batch_size;
    int thread_num = GLB_run_threads;
    int real_max_batch_word_len = 0;

    std::vector<std::vector<Tensor<X86, AK_FLOAT>* >> host_tensor_p_in_list;
    host_tensor_p_in_list=get_slice_input_data(word_idx,1,real_max_batch_word_len,batch_num);

    workers.Reshape("input_0", {real_max_batch_word_len,1,1,1});

    workers.launch();



    for(int i=0; i<host_tensor_p_in_list[0].size(); i++) {
        std::vector<Tensor4dPtr<target_host<X86>::type, AK_FLOAT> > tensor_in_vec(1);
        tensor_in_vec[0]=host_tensor_p_in_list[0][i];
        workers.sync_prediction(tensor_in_vec);
//        auto  d_tensor_p_out_list = workers.sync_prediction(host_tensor_p_in_list);

        // get the output
//        auto d_tensor_p = d_tensor_p_out_list[0];
    }
}


TEST(NetTest, net_execute_base_test) {
    if(GLB_run_mode=="instance"){
        instance_run();
        return;

    }
    if(GLB_run_mode=="worker"){
        worker_run();
        return;
    }
    LOG(ERROR)<<"No support running mode ["<<GLB_run_mode<<"]";
    exit(-1);
}


int main(int argc, const char** argv) {

    Env<X86>::env_init();

    // initial logger
    LOG(INFO) << "argc " << argc;

    if (argc < 3) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/model_test\n \
            anakin_models\n input file\n";
        exit(0);
    } else if (argc >= 3) {
        GLB_model_dir = std::string(argv[1]);
        GLB_input_file = std::string(argv[2]);
    }

    if (argc >= 4) {
        GLB_run_threads = atoi(argv[3]);
    }

    if (argc >= 5) {
        GLB_batch_size = atoi(argv[4]);
    }

    if (argc >= 6) {

        GLB_output_name = std::string(argv[5]);
        if(GLB_output_name=="no_out")
            GLB_output_name="";
    }

    if (argc >= 7) {
        GLB_run_mode = argv[6];
        LOG(INFO)<<"run mode = "<<GLB_run_mode;
    }

    if (argc >= 8) {
        GLB_split_index = atoi(argv[7]);
    }

    if (argc >= 9) {
        GLB_split_word = std::string(argv[8]);
    }



    logger::init(argv[0]);

    for (int i = 0; i < GLB_split_word.size(); i++) {
        LOG(INFO) << "splite  " << i << " in " << GLB_split_word.size() << "  is " << int(char(
                      GLB_split_word[i]));
    }

    //    exit(0);
    //    run_my_test();
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
#else
int main(int argc, const char** argv) {
    return 0;
}

#endif