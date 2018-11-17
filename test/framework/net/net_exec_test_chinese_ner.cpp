
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

int read_file(std::vector<float> &results, const char* file_name) {

    std::ifstream infile(file_name);
    if (!infile.good()) {
        std::cout << "Cannot open " << std::endl;
        return false;
    }
            LOG(INFO)<<"found filename: "<<file_name;
    std::string line;
    while (std::getline(infile, line)) {
        results.push_back((float)atof(line.c_str()));
    }
    return 0;
}
void SplitString(const std::string& s,
                 std::vector<std::string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

int split_word_from_file(
        std::vector<std::vector<float> > &word_idx,
        const std::string input_file_path,
        const std::string split_token,
        const std::string inner_split_token,
        const int col_select) {

    std::ifstream infile(input_file_path.c_str());
    if (!infile.good()) {
        std::cout << "Cannot open " << std::endl;
        return 1;
    }
            LOG(INFO)<<"found filename: "<<input_file_path;
    std::string line;
    std::vector<std::string> split_v;
    std::vector<std::string> split_w;
    int word_count=0;
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
    GLB_word_count=word_count;
    return 0;
}

int get_batch_data_offset(
        std::vector<float> &out_data,
        const std::vector<std::vector<float> > &seq_data,
        std::vector<int> &seq_offset,
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

void anakin_net_thread(std::vector<Tensor4dPtr<X86> > *data_in,
                       std::vector<Tensor4dPtr<X86> > *mention_in,
                       std::string model_path
) {
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    mkl_set_num_threads(1);
    Graph<X86, Precision::FP32> *graph = new Graph<X86, Precision::FP32>();
    //graph = new Graph<Target, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << model_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_path);
    if(!status ) {
         LOG(FATAL) << " [ERROR] " << status.info();
    }
    graph->Reshape("input_0", {GLB_max_word_len, 1, 1, 1});
    graph->Reshape("input_1", {GLB_max_word_len, 1, 1, 1});
    graph->Optimize();
    Net<X86, Precision::FP32> net_executer(*graph, true);
    struct timeval time_start,time_end;

    int slice_10=data_in->size()/10;
    gettimeofday(&time_start, nullptr);
    for (int i = 0; i < data_in->size(); ++i) {
        auto input_tensor=(*data_in)[i];
        auto input_mention_tensor=(*mention_in)[i];
        auto word_in_p = net_executer.get_in("input_0");
        auto mention_in_p = net_executer.get_in("input_1");
        int len_sum=input_tensor->valid_size();
        Shape word_in_shape({len_sum, 1, 1, 1}, Layout_NCHW);
        word_in_p->reshape(word_in_shape);
        for (int j = 0; j < len_sum; ++j) {
            ((float*)word_in_p->mutable_data())[j] = ((float*)input_tensor->data())[j];
        }
        word_in_p->set_seq_offset(input_tensor->get_seq_offset());
        int mention_sum=input_mention_tensor->valid_size();
        Shape mention_in_shape({len_sum, 1, 1, 1}, Layout_NCHW);
        mention_in_p->reshape(mention_in_shape);
        for (int j = 0; j < mention_sum; ++j) {
            ((float*)mention_in_p->mutable_data())[j] = ((float*)input_mention_tensor->data())[j];
        }
        mention_in_p->set_seq_offset(input_mention_tensor->get_seq_offset());
//        timer.start(ctx);
        net_executer.prediction();
//        timer.end(ctx);
        if (i%slice_10==0)
            LOG(INFO)<<"thread run "<<i<<" of "<<data_in->size();
    }
    gettimeofday(&time_end, nullptr);
    float use_ms=(time_end.tv_sec-time_start.tv_sec)*1000.f+(time_end.tv_usec-time_start.tv_usec)/1000.f;
            LOG(INFO)<<"summary_thread :thread total : "<<use_ms<<" ms, avg = "<<(use_ms/data_in->size()/GLB_batch_size);
}

TEST(NetTest, net_execute_base_test) {

    std::vector<std::string> models;
    getModels(GLB_model_dir, models);

    std::vector<std::vector<float> > word_idx;
    std::vector<std::vector<float> > mention_idx;
    if (split_word_from_file(word_idx, GLB_input_file, ";", " ", 1)) {
                LOG(ERROR) << " NOT FOUND " << GLB_input_file;
        exit(-1);
    }
    if (split_word_from_file(mention_idx, GLB_input_file, ";", " ", 3)) {
                LOG(ERROR) << " NOT FOUND " << GLB_input_file;
        exit(-1);
    }

    std::vector<float> word_idx_data;
    std::vector<int> word_seq_offset;
    std::vector<float> mention_idx_data;
    std::vector<int> mention_seq_offset;
    int batch_num =GLB_batch_size;
    int max_batch_word_len=2000;
    int thread_num=GLB_run_threads;

    int real_max_batch_word_len=0;

    std::vector<std::vector<Tensor<X86>* >> host_tensor_p_in_list;
    std::vector<std::vector<Tensor<X86>* >> mention_tensor_p_in_list;
    for(int tid=0;tid<thread_num;++tid){
        std::vector<Tensor<X86>* > data4thread;
        std::vector<Tensor<X86>* > data4thread_m;
        int start_wordid=tid*(word_idx.size()/thread_num);
        int end_wordid=(tid+1)*(word_idx.size()/thread_num);
        for (int i = start_wordid; i < end_wordid; i+=batch_num) {
            int word_len = get_batch_data_offset(word_idx_data, word_idx, word_seq_offset, i, batch_num);
            int mention_len = get_batch_data_offset(mention_idx_data, mention_idx, word_seq_offset, i, batch_num);
            real_max_batch_word_len=real_max_batch_word_len<word_len?word_len:real_max_batch_word_len;
            saber::Shape valid_shape({word_len, 1, 1, 1});
            Tensor4d<X86>* tensor_p=new Tensor4d<X86>(valid_shape);
            CHECK_EQ(word_len,word_idx_data.size())<<"word_len == word_idx_data.size";

            saber::Shape valid_shape_m({word_len, 1, 1, 1});
            Tensor4d<X86>* tensor_m=new Tensor4d<X86>(valid_shape);
            CHECK_EQ(mention_len,mention_idx_data.size())<<"mention_len == mention_idx_data.size";
            for (int j = 0; j < word_idx_data.size(); ++j) {
                ((float*)tensor_p->mutable_data())[j] = word_idx_data[j];
            }
            for (int j = 0; j < mention_idx_data.size(); ++j) {
                ((float*)tensor_m->mutable_data())[j] = mention_idx_data[j];
            }
            tensor_p->set_seq_offset({word_seq_offset});
            data4thread.push_back(tensor_p);
            tensor_m->set_seq_offset({mention_seq_offset});
            data4thread_m.push_back(tensor_m);
        }
        host_tensor_p_in_list.push_back(data4thread);
        mention_tensor_p_in_list.push_back(data4thread_m);
    }
    GLB_max_word_len=real_max_batch_word_len;
    LOG(WARNING) << "Async Runing multi_threads for model: " << models[0]<<",batch dim = "<<batch_num
                 <<",line num = "<<word_idx.size()<<", number of word = "<<GLB_word_count<<",thread number size = "<<thread_num<<",real max = "<<real_max_batch_word_len;

    std::vector<std::unique_ptr<std::thread>> threads;
    struct timeval time_start,time_end;

    gettimeofday(&time_start, nullptr);

    for (int i = 0; i < thread_num; ++i) {
        threads.emplace_back(
                new std::thread(&anakin_net_thread, &host_tensor_p_in_list[i],
                                &mention_tensor_p_in_list[i], models[0]));
    }

    for (int i = 0; i < thread_num; ++i) {
        threads[i]->join();
    }
    gettimeofday(&time_end, nullptr);
    float use_ms=(time_end.tv_sec-time_start.tv_sec)*1000.f+(time_end.tv_usec-time_start.tv_usec)/1000.f;

    LOG(INFO)<<"summary: "<<"thread num = "<<thread_num<<",total time = "<<use_ms<<"ms ,batch = "<<batch_num
             <<",word sum = "<<GLB_word_count<<", seconde/line = "<<(use_ms/word_idx.size())
             <<",QPS = "<<(word_idx.size()/use_ms*1000)<<" line/s , QPS_W = "<<(GLB_word_count/use_ms*1000);
}


int main(int argc, const char** argv){

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
    if(argc>=4){
        GLB_run_threads=atoi(argv[3]);
    }
    if(argc>=4){
        GLB_batch_size=atoi(argv[4]);
    }


    logger::init(argv[0]);

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
#else
int main(int argc, const char** argv){
    return 0;
}

#endif
