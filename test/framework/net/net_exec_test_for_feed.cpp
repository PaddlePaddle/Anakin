#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "debug.h"
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
#elif defined(AMD_GPU)
using Target = AMD;
using Target_H = X86;
#elif defined(USE_MLU)
using Target = MLU;
using Target_H = MLUHX86;
#elif defined(USE_BM_PLACE)
using Target = BM;
using Target_H = BMX86;
#endif

//#define USE_DIEPSE

std::string g_model_path = "/path/to/your/anakin_model";

std::string model_saved_path = g_model_path + ".saved";
int g_batch_size = 0; // 0 means not set max batch
int g_feature_size = 10; // we support different feature size in different slots.
int g_warm_up = 100;
int g_epoch = 1000;
int g_device_id = 0;
int g_thread_num = 1;
std::string g_data_path="";

std::vector<std::string>
        split_string(const std::string& s, char delim) {

    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

void read_slot_file(std::vector<std::vector<float>>& input_data, std::string& data_path, int max_batch = 0) {

    std::ifstream infile(data_path);
    if (!infile.good()) {
        LOG(FATAL) <<"Cannot open " << data_path;
    }
    int max_feature = 0;
    LOG(INFO) << "found filename: " << data_path;
    std::string line;
    int line_num = 0;
    while (std::getline(infile, line)) {
        std::vector<float> line_vector;
        std::vector<std::string> split_line = split_string(line,'\t');
        std::string line_key = split_line[0];
        std::vector<std::string> line_data =
                split_string(split_line[1],' ');
        for (auto c : line_data) {
            line_vector.push_back((float)atof(c.c_str()));
        }
        if (max_feature < line_vector.size()) {
            max_feature = line_vector.size();
        }
        input_data.push_back(line_vector);
        if (max_batch != 0) {
            ++line_num;
            if (line_num >= (412 * max_batch)) {
//                LOG(INFO) << "line_num = " << line_num << " max_batch = " << max_batch;
                break;
            }
        }
    }
    LOG(INFO) << "max_feature = " << max_feature;
}

#if defined(USE_CUDA)||defined(USE_X86_PLACE)
#if defined(USE_X86_PLACE)
#include "mkl_service.h"
#include "omp.h"
#endif

TEST(NetTest, net_execute_base_test) {
#if defined(USE_X86_PLACE)
    if (g_thread_num != 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(g_thread_num);
        mkl_set_num_threads(g_thread_num);
    } else {
        LOG(INFO) << "use all core on CPU!!";
    }
#endif
    std::vector<std::vector<float>> input_data;
    read_slot_file(input_data, g_data_path, g_batch_size);

    CHECK_EQ((input_data.size() % 412), 0) << " FATAL ERROR slot num is not right!!! ";

    std::vector<int> seq_offset{0};
    for (int i = 1; i < input_data.size() + 1; ++i) {
        seq_offset.push_back(seq_offset[i - 1] + input_data[i - 1].size() / 11);
    }
//    printf_pointer(seq_offset.data(), seq_offset.size());

    Graph<Target, Precision::FP32> *graph = new Graph<Target, Precision::FP32>();
    LOG(INFO) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    int total_feature_size = seq_offset[seq_offset.size() - 1]; // this is feature_size
    int slot = 412; // this is slots num
    int max_batch = 2048; // the possible max batch

    // reshape the input_0 's shape for graph model
    Shape shape({max_batch, 1, total_feature_size / max_batch, 11}, Layout_NCHW);

    graph->Reshape("input_0", shape);
//	graph->ResetBatchSize("input_0", g_batch_size);
    LOG(INFO) << "g_batch_size = " << g_batch_size;
    //anakin graph optimization
    graph->Optimize();
    Net<Target, Precision::FP32> net_executer(true);

    net_executer.init(*graph);
    // get in
    auto ins = graph->get_ins();
    auto d_tensor_in_p = net_executer.get_in(ins[0]);
    Shape new_shape({1, 1, total_feature_size, 11}, Layout_NCHW);

    d_tensor_in_p->reshape(new_shape);
    Tensor4d<Target_H> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i = 0; i < valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float *h_data = (float *) (h_tensor_in.mutable_data());

    int idx = 0;
    for (auto i : input_data) {
        for (auto j : i) {
            h_data[idx++] = j;
        }
    }
    d_tensor_in_p->copy_from(h_tensor_in);
    d_tensor_in_p->set_seq_offset({seq_offset});
    // do inference
    Context<Target> ctx(g_device_id, 0, 0);
    saber::SaberTimer<Target> my_time;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";
    // warm up
    for (int i = 0; i < g_warm_up; i++) {
        net_executer.prediction();
    }
    Tensor<Target_H> h_tensor_out;
    h_tensor_out.re_alloc(net_executer.get_out_list()[0]->valid_shape(), AK_FLOAT);

#ifdef ENABLE_OP_TIMER
    net_executer.reset_op_time();
#endif

    my_time.start(ctx);
    //auto start = std::chrono::system_clock::now();
    for (int i = 0; i < g_epoch; i++) {
//        d_tensor_in_p->copy_from(h_tensor_in);
        //DLOG(ERROR) << " g_epoch(" << i << "/" << g_epoch << ") ";
        net_executer.prediction();
//        h_tensor_out.copy_from(*net_executer.get_out_list()[0]);
    }
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif
    my_time.end(ctx);
#ifdef ENABLE_OP_TIMER
    net_executer.print_and_reset_optime_summary(g_epoch);
#endif

    LOG(INFO) << "aveage time " << my_time.get_average_ms() / g_epoch << " ms";
    write_tensorfile(*net_executer.get_out_list()[0], "output.txt");
    //} // inner scope over

    LOG(ERROR) << "inner net exe over !";

    // save the optimized model to disk.
    std::string save_g_model_path = g_model_path + std::string(".saved");
    status = graph->save(save_g_model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    if (!graph) {
        delete graph;
    }
}

#endif

int main(int argc, const char **argv) {
    if (argc < 2) {
    LOG(FATAL) << "no input!!!, usage: ./" << argv[0]
        << " model_path input_data_path [batch_size] [device_id]";
        return -1;
    }
    if (argc > 1) {
        g_model_path = std::string(argv[1]);
    }
    if (argc > 2) {
        g_data_path = std::string(argv[2]);
    }
    if (argc > 3) {
        g_batch_size = atoi(argv[3]);
    }
    if (argc > 4) {
        g_device_id = atoi(argv[4]);
    }
    TargetWrapper<Target>::set_device(g_device_id);
    Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
