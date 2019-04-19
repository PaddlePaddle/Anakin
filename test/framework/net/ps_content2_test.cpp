#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include <vector>
#include "saber/core/tensor_op.h"

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



//#define USE_DIEPSE

//std::string g_model_path = "/home/lxc/projects/models/converter_lego/output/ps.anakin.bin";
std::string g_model_path = "/home/chengyujuan/baidu/sys-hic-gpu/anakin-models/ps/content2.0/content_dnn_2.0.anakin.bin";
//std::string g_model_path = "/home/lxc/projects/anakin_icode/Anakin-2.0/tools/external_converter_v2/output/ps.anakin.bin";
//std::string g_data_path = "/home/lxc/projects/models/test_data/test_40.txt";
//std::string g_data_path = "/home/lxc/projects/models/test_data/fake.txt";
//std::string g_data_path = "/home/lxc/projects/models/test_data/sample_by_query_length.expand.sample_url";
std::string g_data_path = "/home/chengyujuan/ps_content_test_data";
int g_batch_size = 1;
int g_epoch = 1;


std::string model_saved_path = g_model_path + ".saved";

// some data pre-handle funcs.
namespace test_ps {
    std::vector<std::string> input_names{"q_basic_input", "q_bigram0_input", "q_bigram1_input", "pt_basic_input", 
            "pt_bigram0_input", "pt_bigram1_input", "pa_basic_input", "pa_bigram0_input", "pa_bigram1_input"};
    size_t query_len = 50;
    size_t batch_size = 1;
    std::vector<std::string> inputed_lines;
    void set_batch_size (int bs) {batch_size = bs;}

    void load_input_lines(char *filename) {
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

/*
    int string_to_id_buffer(
            float* out_buffer, const int capacity, const std::string& str) {
        std::vector<std::string> id_strs;
        split2(str, id_strs, std::string(" "));
        if ((int)id_strs.size() > capacity){
            fprintf(stderr, "input length(%lu) is larger than capacity(%d)\n",
                    id_strs.size(), capacity);
            return -1;
        }
        for (size_t i = 0; i < id_strs.size(); i++){
            out_buffer[i] = static_cast<float>(atof(id_strs[i].c_str()));
        }
        return id_strs.size();
    }
*/
#ifdef USE_CUDA
    int batch_string_to_input(const std::vector<std::string> &line_vec, Net<Target, Precision::FP32, OpRunType::SYNC>& net_executer){
        
        size_t input_size = input_names.size();
        std::vector<Tensor<Target_H> > h_inputs(input_size);
        std::vector<Tensor<Target>* > d_inputs(input_size);
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
//                split2(number_strs[i][j+1], tmp, std::string(" "));
//                query_shapes[j][0] += tmp.size();
//                add the case that input's empty
                if (number_strs[i][j+1].empty()) {
                    query_shapes[j][0] += 1;
				}else {
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
				}else {
                    split2(number_strs[i][j+1], tmp, std::string(" "));
                    for (size_t i = 0; i < tmp.size(); i++){
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
#endif

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
//                split2(number_strs[i][j+1], tmp, std::string(" "));
//                query_shapes[j][0] += tmp.size();
//                add the case that input's empty
                if (number_strs[i][j+1].empty()) {
                    query_shapes[j][0] += 1;
				}else {
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
				}else {
                    split2(number_strs[i][j+1], tmp, std::string(" "));
                    for (size_t i = 0; i < tmp.size(); i++){
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
#endif

#ifdef USE_CUDA
    int batch_string_to_input(const std::vector<std::string> &line_vec, Net<Target, Precision::FP32>& net_executer){
        
        size_t input_size = input_names.size();
        std::vector<Tensor<Target_H> > h_inputs(input_size);
        std::vector<Tensor<Target>* > d_inputs(input_size);
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
                // split2(number_strs[i][j+1], tmp, std::string(" "));
                // query_shapes[j][0] += tmp.size();
                // add the case that input's empty
                if (number_strs[i][j+1].empty()) {
                    query_shapes[j][0] += 1;
				}else {
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
				}else {
                    split2(number_strs[i][j+1], tmp, std::string(" "));
                    for (size_t i = 0; i < tmp.size(); i++){
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
#endif

// X86
#ifdef USE_X86_PLACE
    int batch_string_to_input(const std::vector<std::string> &line_vec, Net<Target_H, Precision::FP32>& net_executer){
        
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
                // split2(number_strs[i][j+1], tmp, std::string(" "));
                // query_shapes[j][0] += tmp.size();
                // add the case that input's empty
                if (number_strs[i][j+1].empty()) {
                    query_shapes[j][0] += 1;
				}else {
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
				}else {
                    split2(number_strs[i][j+1], tmp, std::string(" "));
                    for (size_t i = 0; i < tmp.size(); i++){
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
#endif

} // namespace test_ps.

#ifdef USE_CUDA
#if 1
TEST(NetTest, net_execute_base_test) {
    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    graph->Optimize();

    // constructs the executer net
	//{ // inner scope
#ifdef USE_DIEPSE
    Net<NV, Precision::FP32, OpRunType::SYNC> net_executer(true);
#else
    Net<NV, Precision::FP32> net_executer(true);
#endif

	net_executer.init(*graph);

    int epoch = 1;
    // do inference
    Context<NV> ctx(0, 0, 0);
    saber::SaberTimer<NV> my_time;
	saber::SaberTimer<NV> my_time1;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";
	// warm up
	/*for(int i=0; i<10; i++) {
		net_executer.prediction();
	}*/
    
    // feed inputs.
    test_ps::load_input_lines(g_data_path.c_str());
	int count = 0;
	float elapsedTime = 0.0f;
	my_time.start(ctx);
    //for (int i = 0; i < test_ps::inputed_lines.size(); i+= test_ps::batch_size) {
    for (int i = 0; i < test_ps::inputed_lines.size() && i < g_epoch * test_ps::batch_size; i+= test_ps::batch_size) {
        std::vector<std::string> line_vec;
        int pre_query_index = -1;
        for (int j = i; j < test_ps::batch_size + i && j < test_ps::inputed_lines.size(); j++) {
            auto line  = test_ps::inputed_lines[j];
            std::vector<std::string> number_strs;
            std::vector<std::string> tmp;
            test_ps::split2(line, number_strs, ";");
            test_ps::split2(number_strs[0], tmp, std::string(" "));
            int cur_query_index = atoi(tmp[0].c_str());
            //LOG(INFO) << "raw str" << line;
            //LOG(INFO) << "pre_query_index:" << pre_query_index;
            //LOG(INFO) << "cur_query_index:" << cur_query_index;
            if (pre_query_index != -1 && cur_query_index != pre_query_index) {                break;
            } else {
                line_vec.push_back(line);
                pre_query_index = cur_query_index;
            }
        }
        i -= (test_ps::batch_size - line_vec.size());
//		LOG(INFO) << "this is line:"<<(i+1);
        int flag = test_ps::batch_string_to_input(line_vec, net_executer);
        if (flag == -1){
            fprintf(stderr,
                "[ERROR]line %d string to input returned error %d\n", i, flag);
            continue;
        }
//		cudaDeviceSynchronize();
            net_executer.prediction();
		//if (count >= 10) {
      	//    my_time1.start(ctx);
		//}
        //for (int k = 0; k< 1000; k++) {
        //    net_executer.prediction();
        //}
		//
		//if (count >=10) {
  		//    my_time1.end(ctx);
  		//    elapsedTime += my_time1.get_average_ms();
		//}
//		cudaDeviceSynchronize();
        auto tensor_out_0_p = net_executer.get_out("qps_out");
        LOG(INFO) << "start: " << i<< " batch_size: "<< line_vec.size();
        test_print(tensor_out_0_p);
		//count++;
		//if (count>=1)
		//	break;
    }
	my_time.end(ctx);
//	LOG(INFO) << "average execute time:" << elapsedTime / (count) << "ms";
    LOG(INFO) << "average execute time:" << elapsedTime / (count-10) << "ms";
//	LOG(INFO) << "average execute time:" << elapsedTime / count << "ms";
	LOG(INFO) << "all execute time:" << my_time.get_average_ms() / (count) << "ms";


    // save the optimized model to disk.
   std::string save_g_model_path = g_model_path + std::string(".saved");
   status = graph->save(save_g_model_path);
   if (!status ) { 
       LOG(FATAL) << " [ERROR] " << status.info(); 
   }
    
    delete graph;
}
#endif 
#endif

#ifdef USE_X86_PLACE
#if 0
TEST(NetTest, net_execute_performance) {
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
#ifdef USE_DIEPSE
    Net<Target_H, Precision::FP32, OpRunType::SYNC> net_executer(true);
#else
    Net<Target_H, Precision::FP32> net_executer(true);
#endif
    net_executer.load_calibrator_config("net_pt_config_x86.txt", "cal_file");
	net_executer.init(*graph);
	    
    // feed inputs.
    test_ps::load_input_lines(g_data_path.c_str());
    for (int i = 0; i < 1/*test_ps::inputed_lines.size()*/; i+= test_ps::batch_size) {
        std::vector<std::string> line_vec;
        for (int j = i; j < test_ps::batch_size + i && j < test_ps::inputed_lines.size(); j++) {
            line_vec.push_back(test_ps::inputed_lines[j]);
        }
		LOG(INFO) << "this is line:"<<(i+1);
        int flag = test_ps::batch_string_to_input(line_vec, net_executer);
        if (flag == -1){
            fprintf(stderr,
                "[ERROR]line %d string to input returned error %d\n", i, flag);
            continue;
        }
		
		// warm up
//		for (int i = 0; i < 50; i++) {
//			net_executer.prediction();
//		}

		int epoch = 1;
//		Context<NV> ctx(0, 0, 0);
		Context<Target_H> ctx;
		saber::SaberTimer<Target_H> my_time;
		LOG(WARNING) << "EXECUTER !!!!!!!! ";
		my_time.start(ctx);	
		for (int i = 0; i < epoch; i++) {
			net_executer.prediction();
		}
		my_time.end(ctx);
		LOG(INFO)<<"average time "<< my_time.get_average_ms()/epoch << " ms";
		auto tensor_out_0_p = net_executer.get_out("qps_out");
        test_print(tensor_out_0_p);
    }

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
        g_epoch = atoi(argv[3]);
    }
    if (argc >= 5){
        g_batch_size = atoi(argv[4]);
    }
    test_ps::set_batch_size(g_batch_size);
    LOG(INFO) << "g_batch_size" << g_batch_size;

    Env<Target>::env_init();
//	Env<Target_H>::env_init();
//	Env<NVHX86>::env_init();
	// initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
