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
#endif

//#define USE_DIEPSE

std::string g_model_path = "/path/to/your/anakin_model";

std::string model_saved_path = g_model_path + ".saved";
int g_batch_size = 1;
int g_warm_up = 10;
int g_epoch = 1000;
int g_device_id = 0;

#ifdef USE_CUDA
#if 1

TEST(NetTest, net_test_load_from_buffer) {
    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    std::ifstream ifs;
    ifs.open (g_model_path, std::ifstream::in);
    if (!ifs.is_open()) {
        LOG(FATAL) << "file open failed";
    }
    ifs.seekg(0, ifs.end);
    int length = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    char * buffer = new char [length];
    ifs.read(buffer, length);
    ifs.close();
    
    // load anakin model files.
    auto status = graph->load(buffer, length);
	if (!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
	}
    graph->ResetBatchSize("input_0", g_batch_size);
    graph->Optimize();
    Net<NV, Precision::FP32> net_executer(true);
    net_executer.init(*graph);
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<Target_H> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    fill_tensor_const(h_tensor_in, 1.f);
    d_tensor_in_p->copy_from(h_tensor_in);
    cudaDeviceSynchronize();
    net_executer.prediction();
    cudaDeviceSynchronize();
    auto h_tensor_out = net_executer.get_out_list()[0];
    LOG(INFO) << "output mean value: " << tensor_mean_value_valid(*h_tensor_out);
    write_tensorfile(*net_executer.get_out_list()[0],"output_b.txt");
}

TEST(NetTest, net_execute_base_test) {
    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    // reshape the input_0 's shape for graph model
    //graph->Reshape("input_0", {1, 8, 640, 640});
	graph->ResetBatchSize("input_0", g_batch_size);

    // register all tensor inside graph
    // graph->RegistAllOut();

    // register edge
    // graph->RegistOut("conv2_2/expand/scale", "relu2_2/expand");
	// graph->RegistOut("relu#3(conv2d_0)","pool2d#4(pool2d_0)");

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
	//{ // inner scope
#ifdef USE_DIEPSE
    //Net<NV, Precision::FP32, OpRunType::SYNC> net_executer(*graph, true);
    Net<NV, Precision::FP32, OpRunType::SYNC> net_executer(true);
#else
    //Net<NV, Precision::FP32> net_executer(*graph, true);
    Net<NV, Precision::FP32> net_executer(true);
#endif

    net_executer.init(*graph);
    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<Target_H> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = (float*)(h_tensor_in.mutable_data());

    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }

    d_tensor_in_p->copy_from(h_tensor_in);

#ifdef USE_DIEPSE
    // for diepse model
    auto d_tensor_in_1_p = net_executer.get_in("input_1");
    Tensor4d<X86> h_tensor_in_1;

    h_tensor_in_1.re_alloc(d_tensor_in_1_p->valid_shape());
    for (int i=0; i<d_tensor_in_1_p->valid_shape().size(); i++) {
        LOG(INFO) << "detect input_1 dims[" << i << "]" << d_tensor_in_1_p->valid_shape()[i];
    }
    h_data = h_tensor_in_1.mutable_data();
    h_data[0] = 1408;
    h_data[1] = 800;
    h_data[2] = 0.733333;
    h_data[3] = 0.733333;
    h_data[4] = 0;
    h_data[5] = 0;
    d_tensor_in_1_p->copy_from(h_tensor_in_1);

    auto d_tensor_in_2_p = net_executer.get_in("input_2");
    Tensor4d<X86> h_tensor_in_2;

    h_tensor_in_2.re_alloc(d_tensor_in_2_p->valid_shape());
    for (int i=0; i<d_tensor_in_2_p->valid_shape().size(); i++) {
        LOG(INFO) << "detect input_2 dims[" << i << "]" << d_tensor_in_2_p->valid_shape()[i];
    }
    h_data = h_tensor_in_2.mutable_data();
    h_data[0] = 2022.56;
    h_data[1] = 989.389;
    h_data[2] = 2014.05;
    h_data[3] = 570.615;
    h_data[4] = 1.489;
    h_data[5] = -0.02;
    d_tensor_in_2_p->copy_from(h_tensor_in_2);
#endif

    //int g_epoch = 1000;
    //int g_warm_up=10;
    // do inference
    Context<NV> ctx(g_device_id, 0, 0);
    saber::SaberTimer<NV> my_time;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";
	// warm up
	for(int i = 0; i < g_warm_up; i++) {
		net_executer.prediction();
	}
    for(auto x:net_executer.get_in_list()){
        fill_tensor_const(*x, 1);
    }
#ifdef ENABLE_OP_TIMER
    net_executer.reset_op_time();
#endif

    my_time.start(ctx);

    //auto start = std::chrono::system_clock::now();
    for(int i = 0; i < g_epoch; i++) {
		//DLOG(ERROR) << " g_epoch(" << i << "/" << g_epoch << ") ";
        net_executer.prediction();
    }
   /* // running part of model
    net_executer.execute_stop_at_node("relu2_2/expand");
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif

	// get inner tensor after stop
    auto tensor_out_inner_p = net_executer.get_tensor_from_edge("conv2_2/expand", "relu2_2/expand");
    LOG(WARNING) << "inner tensor avg value : " << tensor_average(tensor_out_inner_p);
#ifdef USE_CUDA
	cudaDeviceSynchronize();
#endif

    for (int i = 0; i < 3; i++) {
    	net_executer.execute_start_from_node("relu2_2/expand");
    }

#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif*/

    //auto end = std::chrono::system_clock::now();

    //double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //LOG(WARNING) << "avg time : " << time/g_epoch <<" ms";
    cudaDeviceSynchronize();
    my_time.end(ctx);
#ifdef ENABLE_OP_TIMER
    net_executer.print_and_reset_optime_summary(g_epoch);
#endif

    LOG(INFO)<<"aveage time "<<my_time.get_average_ms()/g_epoch << " ms";
    write_tensorfile(*net_executer.get_out_list()[0],"output.txt");
	//} // inner scope over

	LOG(ERROR) << "inner net exe over !";
    //for (auto x:net_executer.get_out_list()){
    //    print_tensor(*x);
    //}
    //auto& tensor_out_inner_p = net_executer.get_tensor_from_edge("data_perm", "conv1");
	

    // get out yolo_v2
    /*auto tensor_out_0_p = net_executer.get_out("loc_pred_out");
    auto tensor_out_1_p = net_executer.get_out("obj_pred_out");
    auto tensor_out_2_p = net_executer.get_out("cls_pred_out");
    auto tensor_out_3_p = net_executer.get_out("ori_pred_out");
    auto tensor_out_4_p = net_executer.get_out("dim_pred_out");*/

	// get outs cnn_seg 
	/*auto tensor_out_0_p = net_executer.get_out("slice_[dump, mask]_out");
	auto tensor_out_1_p = net_executer.get_out("category_score_out");
	auto tensor_out_2_p = net_executer.get_out("instance_pt_out");
   	auto tensor_out_3_p = net_executer.get_out("confidence_score_out");
	auto tensor_out_4_p = net_executer.get_out("class_score_out");
	auto tensor_out_5_p = net_executer.get_out("heading_pt_out");
	auto tensor_out_6_p = net_executer.get_out("height_pt_out");*/

	// restnet 101
 	//auto tensor_out_0_p = net_executer.get_out("elementwise_add_0.tmp_0_out");
	//auto tensor_out_0_p = net_executer.get_out("prob_out");

	//auto tensor_out_0_p = net_executer.get_out("detection_output_0.tmp_0_out");

    // get out result
    //LOG(WARNING)<< "result avg: " << tensor_average(tensor_out_0_p);
	//test_print(tensor_out_0_p);

    // mobilenet-v2
	//auto tensor_out_0_p = net_executer.get_out("dim_pred_out");


    // get out result
    //LOG(WARNING)<< "result avg: " << tensor_average(tensor_out_0_p);
	//test_print(tensor_out_0_p);



    // save the optimized model to disk.
    std::string save_g_model_path = g_model_path + std::string(".saved");
    status = graph->save(save_g_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }
    if (!graph){
        delete graph;
    }
}
#endif 
#endif

#if 0
TEST(NetTest, net_execute_reconstruction_test) {
    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from optimized model " << model_saved_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_saved_path);
    if (!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    // regisiter output tensor
    //graph->RegistOut("data_perm",  "data_scale");
    //graph->RegistOut("data_perm",  "conv1");

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
    Net<NV, Precision::FP32> net_executer(*graph);

    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<X86> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = h_tensor_in.mutable_data();

    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }

    d_tensor_in_p->copy_from(h_tensor_in);

    // do inference
    Context<NV> ctx(g_device_id, 0, 0);
    saber::SaberTimer<NV> my_time;
    my_time.start(ctx);

    LOG(WARNING) << "EXECUTER !!!!!!!! ";
    for (int i=0; i<1; i++) {
        net_executer.prediction();

    }
    my_time.end(ctx);
    LOG(INFO)<<"aveage time "<<my_time.get_average_ms()/1 << " ms";

    //auto tensor_out_inner_p = net_executer.get_tensor_from_edge("data_perm",  "conv1");

    // get out
    /*auto tensor_out_0_p = net_executer.get_out("loc_pred_out");
    auto tensor_out_1_p = net_executer.get_out("obj_pred_out");
    auto tensor_out_2_p = net_executer.get_out("cls_pred_out");
    auto tensor_out_3_p = net_executer.get_out("ori_pred_out");
    auto tensor_out_4_p = net_executer.get_out("dim_pred_out");*/

    
    auto tensor_out_0_p = net_executer.get_out("dim_pred_out");


    // get out result
	test_print(tensor_out_0_p);

}
#endif

int main(int argc, const char** argv){
    if (argc < 2){
        LOG(ERROR) << "no input!!!, usage: ./" << argv[0] << " model_path [batch size] [warm_up_iter] [test_iter] [device_id]";
        return -1;
    }
    if (argc > 1) {
        g_model_path = std::string(argv[1]);
    }
    if (argc > 2) {
        g_batch_size = atoi(argv[2]);
    }
    if (argc > 3) {
        g_warm_up = atoi(argv[3]);
    }
    if (argc > 4) {
        g_epoch = atoi(argv[4]);
    }
    if (argc > 5) {
        g_device_id = atoi(argv[5]);
    }
    TargetWrapper<Target>::set_device(g_device_id);
    Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
