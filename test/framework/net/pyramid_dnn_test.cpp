#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include <vector>
#include "saber/core/tensor_op.h"
#include <omp.h>

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
#endif

std::string g_model_path = "./ps_shared.anakin.bin";
std::string g_data_path = "/home/chaowen/4u8/cuichaowen/backup/ps_anakin/sample_by_query_length.expand.sample_url";
int g_epoch = 1;
int g_num_threads = 1;
int g_batch_size = 1;


std::string g_model_saved_path = g_model_path + ".saved";

// some data pre-handle funcs.
namespace test_ps {
    std::vector<std::string> input_names{"qb_input", "qp_input", "p_tb_input", "p_tp_input"};
    size_t query_len = 50;
    size_t batch_size = g_batch_size;
    std::vector<std::string> inputed_lines;
    void set_batch_size(int batch_size_in) {
         batch_size = batch_size_in;
    }

    void load_input_lines(const char *filename) {
        static const int max_line_buf_size = 100 * 1024 * 1024;
        char *line_buffer = (char *)calloc(max_line_buf_size, sizeof(char));
        FILE *input_file = fopen(filename, "r");

        while (fgets(line_buffer, max_line_buf_size, input_file)) {
            // trim newline at end
            char *pos = NULL;
            if ((pos = strchr(line_buffer, '\n')) != NULL){
                *pos = 0;
            }
            inputed_lines.push_back(line_buffer);
        }
        free(line_buffer);
        line_buffer = NULL;
        fclose(input_file);
    }

    void split2(
            const std::string& main_str,
            std::vector<std::string>& str_list,
            const std::string & delimiter) {
        size_t pre_pos = 0;
        size_t position = 0;
        std::string tmp_str;

        str_list.clear();
        if (main_str.empty()) {
            return;
        }

        while ((position = main_str.find(delimiter, pre_pos)) != std::string::npos) {
            tmp_str.assign(main_str, pre_pos, position - pre_pos);
            str_list.push_back(tmp_str);
            pre_pos = position + 1;
        }

        tmp_str.assign(main_str, pre_pos, main_str.length() - pre_pos);

        if (!tmp_str.empty()) {
            str_list.push_back(tmp_str);
        }
    }

#ifdef USE_X86_PLACE
int batch_string_to_input(const std::vector<std::string> &line_vec, Net<Target_H, Precision::FP32, OpRunType::SYNC>& net_executer){
        
        size_t input_size = input_names.size();
        std::vector<Tensor<Target_H> > h_inputs(input_size);
        std::vector<Tensor<Target_H>* > d_inputs(input_size);
        for (size_t i = 0; i < input_size; i++) {
            d_inputs[i] = net_executer.get_in(input_names[i]);
        }
        
        std::vector<std::vector<int> > offset;
        offset.resize(input_size);
        int batch = line_vec.size();
        for (size_t i = 0; i < input_size; i++) {
            offset[i].resize(batch + 1);
            offset[i][0] = 0;
        }
        // determin inputs' shape.
        std::vector<std::vector<std::string>> number_strs(line_vec.size());
        std::vector<Shape> query_shapes(input_size);
        for (size_t i = 0; i < input_size; i++) {
            query_shapes[i][0] = 0;
            query_shapes[i][1] = 1;
            query_shapes[i][2] = 1;
            query_shapes[i][3] = 1;
        }

        for (size_t i = 0; i < line_vec.size(); i++) {
            split2(line_vec[i], number_strs[i], ";");
            if (number_strs[i].size() < input_size + 1){
                fprintf(stderr, "input slots is no enough, has %lu expect %lu",
                        number_strs[i].size(), input_size);
                return -1;
            }
            std::vector<std::string> tmp;
            for (size_t j = 0; j < input_size; j++) {
                if (number_strs[i][j+1].empty()) {
                    query_shapes[j][0] += 1;
				} else {
                    split2(number_strs[i][j+1], tmp, std::string(" "));
					query_shapes[j][0] += tmp.size();
				}
                offset[j][i+1] = query_shapes[j][0];
            }
        }

        //reshape
        for (size_t i = 0; i < input_size; i++) {
            h_inputs[i].reshape(query_shapes[i]);
            d_inputs[i]->reshape(query_shapes[i]);
        }
        // feed inputs.
        for (size_t i = 0; i < line_vec.size(); i++) {
            std::vector<std::string> tmp;
            for (size_t j = 0; j < input_size; j++) {
                float* h_data = (float*)h_inputs[j].mutable_data();
                h_data = h_data + offset[j][i];
				if (number_strs[i][j+1].empty()) {
                    h_data[0] = -1; //padding_idx == -1.
				} else {
                    split2(number_strs[i][j+1], tmp, std::string(" "));
                    for (size_t i = 0; i < tmp.size(); i++) {
                        h_data[i] = static_cast<float>(atof(tmp[i].c_str()));
                    }
				}
            }
        }
        for (size_t i = 0; i < input_size; i++) {
            d_inputs[i]->copy_from(h_inputs[i]);
            d_inputs[i]->set_seq_offset({offset[i]});
        }

        return 0;
    }
// X86
    int batch_string_to_input(const std::vector<std::string> &line_vec, Net<Target_H, Precision::FP32>& net_executer){
        int max_length = 100;
        size_t input_size = input_names.size();
        std::vector<Tensor<Target_H>* > d_inputs(input_size);
        for (size_t i = 0; i < input_size; i++) {
            d_inputs[i] = net_executer.get_in(input_names[i]);
            d_inputs[i]->reshape(Shape({test_ps::batch_size * max_length, 1, 1, 1}, Layout_NCHW));
        }
        
        std::vector<std::vector<int> > offset;
        offset.resize(input_size);
        int batch = line_vec.size();
        for (size_t i = 0; i < input_size; i++) {
            offset[i].resize(batch + 1);
            offset[i][0] = 0;
        }
        // determin inputs' shape.
        std::vector<std::vector<std::string>> number_strs(line_vec.size());
        Shape temp({0, 0, 0, 0});
        std::vector<Shape> query_shapes(input_size, temp);
        // for (size_t i = 0; i < input_size; i++) {
        //     query_shapes[i]({0, 0, 0, 0});
        // }
        for (size_t i = 0; i < input_size; i++) {
            query_shapes[i][0] = 0;
            query_shapes[i][1] = 1;
            query_shapes[i][2] = 1;
            query_shapes[i][3] = 1;
        }

        for (size_t i = 0; i < line_vec.size(); i++) {
            split2(line_vec[i], number_strs[i], ";");
            if (number_strs[i].size() < input_size + 1){
                fprintf(stderr, "input slots is no enough, has %lu expect %lu",
                        number_strs[i].size(), input_size);
                return -1;
            }
            std::vector<std::string> tmp;
            for (size_t j = 0; j < input_size; j++) {
                // add the case that input's empty
                if (number_strs[i][j+1].empty()) {
                    query_shapes[j][0] += 1;
	        } else {
                    split2(number_strs[i][j+1], tmp, std::string(" "));
		    query_shapes[j][0] += tmp.size();
                }
                float* h_data = (float*)d_inputs[j]->mutable_data();
                h_data = h_data + offset[j][i];
                if (number_strs[i][j+1].empty()) {
                    h_data[0] = -1; //padding_idx == -1.
                } else {
                    split2(number_strs[i][j+1], tmp, std::string(" "));
                    for (size_t k = 0; k < tmp.size(); k++){
                        h_data[k] = static_cast<float>(atof(tmp[k].c_str()));
                    }
                }
                offset[j][i+1] = query_shapes[j][0];
            }
        }

        //reshape
        for (size_t i = 0; i < input_size; i++) {
            d_inputs[i]->reshape(query_shapes[i]);
            d_inputs[i]->set_seq_offset({offset[i]});
        }

        return 0;
    }
#endif
} // namespace test_ps.

#ifdef USE_X86_PLACE
#if 1
TEST(NetTest, net_execute_performance) {
    omp_set_num_threads(g_num_threads);
    Graph<Target_H, Precision::FP32>* graph = new Graph<Target_H, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    graph->Optimize();

    // constructs the executer net
	//{ // inner scope
    Net<Target_H, Precision::FP32> net_executer(true);
    //net_executer.load_calibrator_config("net_pt_config_x86.txt", "cal_file");
	net_executer.init(*graph);
	    
    // feed inputs.
    test_ps::load_input_lines(g_data_path.c_str());
    int batch_num = g_epoch * test_ps::batch_size;
	Context<Target_H> ctx;
	saber::SaberTimer<Target_H> my_time;
	LOG(WARNING) << "EXECUTER !!!!!!!! ";
#ifdef ENABLE_OP_TIMER
    net_executer.reset_op_time();
#endif
	my_time.start(ctx);	
    for (int i = 0; i < batch_num/*test_ps::inputed_lines.size()*/; i+= test_ps::batch_size) {
        std::vector<std::string> line_vec;
        int start = i % test_ps::inputed_lines.size();
        for (int j = start; j < test_ps::batch_size + start && j < test_ps::inputed_lines.size(); j++) {
            line_vec.push_back(test_ps::inputed_lines[j]);
        }
		//LOG(INFO) << "this is line:"<<(i+1);
        int flag = test_ps::batch_string_to_input(line_vec, net_executer);
        if (flag == -1){
            fprintf(stderr,
                "[ERROR]line %d string to input returned error %d\n", i, flag);
            continue;
        }
		
		//int epoch = 1;
//		Context<NV> ctx(0, 0, 0);
	    net_executer.prediction();
            //auto tensor_out_0_p = net_executer.get_out("ps_out");
            //test_print(tensor_out_0_p);
    }
	my_time.end(ctx);
#ifdef ENABLE_OP_TIMER
    net_executer.print_and_reset_optime_summary(g_epoch);
#endif
	LOG(INFO)<<"average time "<< my_time.get_average_ms()/g_epoch << " ms";

	delete graph;
}
#endif
#endif
int main(int argc, const char** argv){
    if (argc >=2) {
        g_model_path = argv[1];
    }  
    if (argc >= 3){
        g_data_path = argv[2];
    } 
    if (argc >= 4){
        g_num_threads = atoi(argv[3]);
    } 
    if (argc >= 5){
        g_epoch = atoi(argv[4]);
    } 
    if (argc >= 6){
        g_batch_size = atoi(argv[5]);
    }
    test_ps::set_batch_size(g_batch_size);
    LOG(INFO) << "g_batch_size" << g_batch_size; 
    Env<Target>::env_init();
	// initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
