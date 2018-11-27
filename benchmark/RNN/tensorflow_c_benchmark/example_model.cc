#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "vector"
#include <fstream>
#include <thread>
#include "sys/time.h"
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

using namespace tensorflow;
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
std::vector<std::vector<float> > get_input_data() {
    std::vector<std::vector<float> > word_idx;

    if (split_word_from_file(word_idx, GLB_input_file, GLB_split_word, " ", GLB_split_index)) {
        LOG(ERROR) << " NOT FOUND " << GLB_input_file;
        exit(-1);
    }

    return word_idx;
};
void sess_thread(std::vector<tensorflow::Tensor*>* tensor_vec) {
    SessionOptions opts;
    opts.config.set_intra_op_parallelism_threads(1);
    opts.config.set_inter_op_parallelism_threads(1);
    opts.config.set_use_per_session_threads(true);
    Session* session;
    Status status = NewSession(opts, &session);

    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return ;
    } else {
        std::cout << "Session created successfully" << std::endl;
    }

    // Load the protobuf graph
    GraphDef graph_def;
    std::string graph_path = GLB_model_dir;//argv[1];
    status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);

    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return ;
    } else {
        std::cout << "Load graph protobuf successfully" << std::endl;
    }

    // Add the graph to the session
    status = session->Create(graph_def);

    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return ;
    } else {
        std::cout << "Add graph to session successfully" << std::endl;
    }

    {
        //warm up
        std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            { "x_input", *(*tensor_vec)[0] },
        };
        std::vector<tensorflow::Tensor> outputs;
        session->Run(inputs, {"Softmax"}, {}, &outputs);

        if (!status.ok()) {
            std::cerr << status.ToString() << std::endl;
            return ;
        } else {
            //  std::cout << "Run session successfully i" << std::endl;
        }


    }

    std::cout << "thread ready to run " << std::endl;
    struct timeval time_start, time_end;

    gettimeofday(&time_start, nullptr);
    {
        for (int i = 0; i < tensor_vec->size(); i++) {
            std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
                { "x_input", *(*tensor_vec)[i] },
            };
            std::vector<tensorflow::Tensor> outputs;
            session->Run(inputs, {"Softmax"}, {}, &outputs);

            if (!status.ok()) {
                std::cerr << status.ToString() << std::endl;
                return ;
            } else {
                //  std::cout << "Run session successfully i" << std::endl;
            }
        }


    }
    gettimeofday(&time_end, nullptr);

    float use_ms = (time_end.tv_sec - time_start.tv_sec) * 1000.f + (time_end.tv_usec -
                   time_start.tv_usec) / 1000.f;
    std::cout << "thread summary : " << "usetime = " << use_ms << " ms," << "word_sum = " <<
              GLB_word_count << ",delay = " << (use_ms / tensor_vec->size()) << ", QPS = " <<
              (GLB_word_count / use_ms * 1000) << std::endl;

    session->Close();
}
/**
 * @brief deep model for click through rate prediction
 * @details [long description]
 *
 * @param argv[1] graph protobuf
 *
 * @return [description]
 */
int main(int argc, char* argv[]) {
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

    // Initialize a tensorflow session

    std::vector<std::vector<float> > word_idx;
    word_idx = get_input_data();
    std::vector<tensorflow::Tensor*> tensor_vec;

    for (int i = 0; i < word_idx.size(); i++) {
        tensorflow::Tensor* t_tensor_p = new Tensor(DT_INT32, TensorShape({1, word_idx[i].size()}));
        auto input_tensor_mapped = t_tensor_p->tensor<int, 2>();

        for (int j = 0; j < word_idx[i].size(); j++) {
            input_tensor_mapped(0, j) = word_idx[i][j];

        }

        tensor_vec.push_back(t_tensor_p);
    }

    std::cout << "get word success!" << std::endl;
    std::cout << "first data = " << tensor_vec[0]->tensor<int, 2>()(0, 0) << std::endl;
    // Setup inputs and outputs:
    // Our graph doesn't require any inputs, since it specifies default values,
    // but we'll change an input to demonstrate.
    std::vector<std::unique_ptr<std::thread>> threads;
    int thread_num = GLB_run_threads;

    for (int i = 0; i < thread_num; ++i) {
        threads.emplace_back(
            new std::thread(&sess_thread, &tensor_vec));
    }

    for (int i = 0; i < thread_num; ++i) {
        threads[i]->join();
    }

    // Grab the first output (we only evaluated one graph node: "c")
    // and convert the node to a scalar representation.
    //auto output_c = outputs[0].scalar<float>();

    // (There are similar methods for vectors and matrices here:
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

    // Print the results
    //std::cout << outputs[0].DebugString() << std::endl; // Tensor<type: float shape: [] values: 30>
    //std::cout << "output value: " << output_c() << std::endl; // 30

    // Free any resources used by the session

    return 0;
}
