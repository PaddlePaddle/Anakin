#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include "saber/core/tensor_op.h"
//#include "saber/funcs/impl/mlu/mlu_helper.h"
#include <chrono>
#include <fstream>
#include <cassert>

#if defined(USE_CUDA)
using Target = NV;
using Target_H = NVHX86;
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
#elif defined(USE_BM)
using Target = BM;
using Target_H = BMX86;
#endif

bool g_fusion = true;
int g_model_parallel = 8;
int g_batch_size = 1;
//#define USE_DIEPSE

std::string model_path =
    "/home/cuichaowen/github_anakin/Anakin/build/yolo_camera_detector.anakin.bin";
#ifdef USE_MLU
std::string model_saved_path = model_path + ".saved";

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c) {
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

TEST(NetTest, net_execute_base_test) {

#ifdef USE_MLU
    std::shared_ptr<Context<MLU>> ctx1 = std::make_shared<Context<MLU>>(0, 0, 0);
    //set parallel.
    ctx1->set_model_parallel(g_model_parallel);
    ctx1->set_fusion(g_fusion);

    if (g_fusion) {
        LOG(INFO) << "MLU is using fusion mode....";
    } else {
        LOG(INFO) << "MLU is usiing no_fusion mode....";
    }

    Graph<MLU, Precision::FP32>* graph = new Graph<MLU, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << model_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_path);
    LOG(INFO) << "load succeed";

    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    auto input_names = graph->get_ins();

    for (auto input : input_names) {
        graph->ResetBatchSize(input, g_batch_size);
    }

    LOG(INFO) << "graph Optimize";
    graph->fusion_optimize(true);
    LOG(INFO) << "after graph Optimize";
    //    Net<MLU, Precision::FP32, OpRunType::SYNC> net_executer(*graph, ctx1, true);
    Net<MLU, Precision::FP32, OpRunType::SYNC> net_executer;
    net_executer.fusion_init(*graph, ctx1, true);
    // get in
    Tensor<Target_H> h_tensor_in;

    for (auto input : input_names) {
        auto d_tensor_in_p = net_executer.get_in(input);
        auto shape_in = d_tensor_in_p->valid_shape();

        for (int i = 0; i < shape_in.size(); i++) {
            LOG(INFO) << "detect input_0 dims[" << i << "]" << shape_in[i];
        }

        h_tensor_in.re_alloc(shape_in);
        float* h_data = (float*)h_tensor_in.mutable_data();

        for (int i = 0; i < h_tensor_in.valid_size(); i++) {
            h_data[i] = 1.0f;
        }

        d_tensor_in_p->copy_from(h_tensor_in);
    }

    int epoch = 10;
    // do inference
    Context<MLU> ctx(0, 0, 0);
    saber::SaberTimer<MLU> my_time;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";

    // warm up
    for (int i = 0; i < 10; i++) {
        net_executer.fusion_prediction();
    }


#ifdef ENABLE_OP_TIMER
    net_executer.reset_op_time();
#endif

    my_time.start(ctx);

    //auto start = std::chrono::system_clock::now();
    for (int i = 0; i < epoch; i++) {
        //DLOG(ERROR) << " epoch(" << i << "/" << epoch << ") ";
        net_executer.fusion_prediction();
    }

    my_time.end(ctx);

    std::ofstream file1;
    std::string output_file1 = model_path;
    std::vector<std::string> split_output_file1;
    std::cout << "output_file: " << output_file1 << std::endl;
    std::string token1 = "/";
    SplitString(output_file1, split_output_file1, token1);

    for (auto str : split_output_file1) {
        std::cout << "--:" << str << std::endl;
    }

    int index_file1 = split_output_file1.size() - 1;
    output_file1 = std::string(split_output_file1[index_file1]);
    std::vector<std::string> output_file_new1;
    token1 = ".";
    SplitString(output_file1, output_file_new1, token1);
    //    output_file1 = output_file_new1[0];
    //    output_file1 += "_output.txt";
    //    file1.open(output_file1);

    //Get output
    Tensor<Target_H> h_tensor_out;
    auto output_name = graph->get_outs();
    LOG(INFO) << "output counts of this graph: " << output_name.size();

    for (int i = 0; i < output_name.size(); i++) {
        output_file1 = output_file_new1[0] + "_out_" + std::to_string(i) + ".txt";
        file1.open(output_file1);
        assert(file1.is_open());
        auto tensor_out_p = net_executer.get_out(output_name[i]);
        auto shape_out = tensor_out_p->valid_shape();
        h_tensor_out.re_alloc(shape_out);
        h_tensor_out.copy_from(*tensor_out_p);
        float* data = (float*)h_tensor_out.data();

        //write shape
        for (int i = 0; i < shape_out.size(); i++) {
            file1 << shape_out[i] << " ";
        }

        file1 << std::endl;

        for (int i = 0; i < h_tensor_out.valid_size(); i++) {
            file1  << data[i] << std::endl;
        }

        file1.close();
    }

#ifdef ENABLE_OP_TIMER
    std::ofstream file;
    std::string output_file = model_path;
    std::vector<std::string> split_output_file;
    std::cout << "output_file: " << output_file << std::endl;
    std::string token = "/";
    SplitString(output_file, split_output_file, token);

    for (auto str : split_output_file) {
        std::cout << "--:" << str << std::endl;
    }

    int index_file = split_output_file.size() - 1;
    output_file = std::string(split_output_file[index_file]);
    std::vector<std::string> output_file_new;
    token = ".";
    SplitString(output_file, output_file_new, token);
    output_file = output_file_new[0];
    output_file += ".csv";
    std::cout << "output_file: " << output_file << std::endl;
    file.open(output_file);
    std::vector<float> op_time = net_executer.get_op_time();
    LOG(INFO) << "op_time size:" << op_time.size();
    auto exec_funcs = net_executer.get_exec_funcs();
    auto op_param = net_executer.get_op_param();
    auto op_input_shape = net_executer.get_op_input_shape();
    LOG(INFO) << "exec_funs:" << exec_funcs.size();
    LOG(INFO) << "op_param:" << op_param.size();
    LOG(INFO) << "op_input_shape:" << op_input_shape.size();
    float time_all = 0.0f;
    file << "op_type,filter_Num,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,group,input_N,input_C,intput_H,input_W,time"
         << std::endl;

    for (int i = 0; i <  op_time.size(); i++) {
        file << exec_funcs[i].op_name << "," << op_param[i];

        if (op_input_shape[i].size() >= 4) {
            file << op_input_shape[i][0] << ","
                 << op_input_shape[i][1] << ","
                 << op_input_shape[i][2] << ","
                 << op_input_shape[i][3] << ",";
        } else {
            file << -1 << ","
                 << -1 << ","
                 << -1 << ","
                 << -1 << ",";
        }

        double times_op = op_time[i] / epoch;
        time_all += times_op;
        file << times_op << ",\n";
    }

    //         file.close();
    file << "Sum up, \n";
    file << "op_type,filter_Num,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,group,time,ratio"
         << std::endl;
    std::map<std::string, float> op_map;
    std::map<std::string, std::string> op_type_map;

    for (int i = 0; i < op_time.size(); i++) {
        std::string str_tmp = exec_funcs[i].op_name + std::string(",") + op_param[i];
        auto it = op_map.find(str_tmp);

        if (it != op_map.end()) {
            op_map[str_tmp] += op_time[i];
        } else {
            op_map.insert(std::pair<std::string, float>(str_tmp, op_time[i]));
            //                 op_type_map.insert(std::pair<std::string, std::string>(op_param[i], exec_funcs[i].op_name));
        }
    }

    for (auto it = op_map.begin(); it != op_map.end(); ++it) {
        //             LOG(INFO)  << " "<<it->first << "  " << (it->second) / epoch << " ms";
        float tmp_time = (it->second) / epoch;
        file << it->first  << tmp_time << "," << tmp_time / time_all * 100 << "%" << std::endl;
    }

    file.close();

#endif
    // wirte to file
    std::string res_file = "res.csv";
    std::fstream _file_exist;
    _file_exist.open(res_file, std::ios::in);

    if (_file_exist) {
        file1.open(res_file, std::ios::app);
        file1 << model_path << "," << output_file_new1[0] << "," << g_batch_size << "," <<
              my_time.get_average_ms() / epoch << std::endl;
        file1.close();
    } else {
        file1.open(res_file);
        file1 << "model_path, model_name, batch_size, time" << std::endl;
        file1 << model_path << "," << output_file_new1[0] << "," << g_batch_size << "," <<
              my_time.get_average_ms() / epoch << std::endl;
        file1.close();
    }

    if (_file_exist) {
        _file_exist.close();
    }

    LOG(INFO) << "model: " << model_path << ",batch_size: " << g_batch_size << ",average time: " <<
              my_time.get_average_ms() / epoch << " ms";
    //    LOG(INFO) << "model" << model_path << " OK!";
    //    LOG(ERROR) << "inner net exe over !";
    delete graph;
    LOG(INFO) << "end of test.";
#endif
}

int main(int argc, const char** argv) {
    if (argc > 1) {
        model_path = argv[1];
    }

    if (argc > 2) {
        g_model_parallel = atoi(argv[2]);
    }

    if (argc > 3) {
        g_batch_size = atoi(argv[3]);
    }

    Env<Target>::env_init();
    //    Env<Target_H >::env_init();
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    LOG(INFO) << "before free env";
    Env<Target>::env_exit();
    LOG(INFO) << "after free env";
    return 0;
}
#else
int main(int argc, const char** argv) {
    return 0;
}
#endif

