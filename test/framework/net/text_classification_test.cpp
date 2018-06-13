
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

#define DEFINE_GLOBAL(type, var, value) \
        type (GLB_##var) = (value)
DEFINE_GLOBAL(std::string, model_dir, "");
DEFINE_GLOBAL(std::string, input_file, "");
#ifdef USE_CUDA
using Target = NV;
#endif
#ifdef USE_X86_PLACE
using Target = X86;
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
        }
        word_idx.push_back(word);
    }
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
        }
        seq_offset.push_back(len);
    }
    return len;
}

TEST(NetTest, chinese_ner_executor) {

    std::vector<std::string> models;
    getModels(GLB_model_dir, models);
    std::vector<std::vector<float> > word_idx;
    if (split_word_from_file(word_idx, GLB_input_file, "\t", " ", 0)) {
        LOG(ERROR) << " NOT FOUND " << GLB_input_file;
        exit(-1);
    }
    LOG(INFO) << "READ SUCCESS!! I got " << word_idx.size() << " records";
    std::vector<float> word_idx_data;
    std::vector<int> word_seq_offset;
    int batch_num = 6;

    graph = new Graph<Target, AK_FLOAT, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << models[0] << " ...";
    // load anakin model files.
    auto status = graph->load(models[0]);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    graph->Reshape("input_0", {1000, 1, 1, 1});
    //anakin graph optimization
    graph->Optimize();
    Net<Target, AK_FLOAT, Precision::FP32> net_executer(*graph, true);
    SaberTimer<X86> timer;
    Context<X86> ctx;

//    for (int i = 0; i < word_idx.size(); i += batch_num) {
    {
#if 1
        int i = 0;
        //int word_len = get_batch_data_offset(word_idx_data, word_idx, word_seq_offset, i, batch_num);
        int word_len = 7;
        word_idx_data = {20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0};
        word_seq_offset = {0, 5, 7};
        for(int i=0;i<word_seq_offset.size();i++){
            LOG(INFO)<<"seq offset ["<<i<<"] = "<<word_seq_offset[i];
        }
        LOG(INFO)<<"word len = "<<word_len<<",word_idx_data = "<<word_idx_data.size();
        auto word_in_p = net_executer.get_in("input_0");
        word_in_p->reshape({word_len, 1, 1, 1});

        for (int j = 0; j < word_idx_data.size(); ++j) {
            word_in_p->mutable_data()[j] = word_idx_data[j];
//            word_in_p->mutable_data()[j] = j % 5000;
        }

        word_in_p->set_seq_offset(word_seq_offset);
#endif
//        auto word_in_p = net_executer.get_in("input_0");
//        word_in_p->mutable_data()[0]=1;
//        word_in_p->mutable_data()[1]=2;
//        word_in_p->mutable_data()[2]=3;
//        word_in_p->set_seq_offset({0,3});

//        timer.start(ctx);
        net_executer.prediction();
//        timer.end(ctx);


//        auto tensor_out_5_p = net_executer.get_out("fc_2.tmp_2_out");
//        int v_size = tensor_out_5_p->valid_size();
//        for (int j = 0; j < v_size; ++j) {
//            std::cout << tensor_out_5_p->data()[j]<<" ";
//        }
//        std::cout << std::endl;
    }
    //LOG(INFO)<<"elapse time: "<<timer.get_average_ms()<<" ms";
}

int main(int argc, const char** argv) {
    // initial logger
    LOG(INFO) << "argc " << argc;

    if (argc < 3) {
        LOG(INFO) << "Example of Usage:\n \
        ./output/unit_test/model_test\n \
            anakin_models\n input file\n";
        exit(0);
    } else if (argc == 3) {
        GLB_model_dir = std::string(argv[1]);
        GLB_input_file = std::string(argv[2]);
    }
//    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
