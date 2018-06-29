#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>

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

//#define USE_DIEPSE

//std::string model_path = "/home/chaowen/anakin_v2/model_v2/anakin-models/adu/anakin_models/diepsie_light_head/diepsie_light_head.anakin.bin";

//std::string model_path = "/home/chaowen/anakin_v2/model_v2/anakin-models/adu/anakin_models/diepsie_light_head/diepsie_light_head_base.anakin.bin";


//std::string model_path = "/home/chaowen/anakin_v2/model_v2/anakin-models/adu/anakin_models/diepsie_light_head/densebox.anakin.bin";

//std::string model_path = "/home/chaowen/anakin_v2/model_v2/anakin-models/adu/anakin_models/diepsie_light_head/cnn_seg.anakin.bin";

//std::string model_path = "/home/chaowen/anakin_v2/model_v2/anakin-models/adu/anakin_models/diepsie_light_head/yolo_camera_detector.anakin.bin";

//std::string model_path = "/home/chaowen/anakin_v2/model_v2/anakin-models/adu/anakin_models/diepsie_light_head/yolo_lane_v2.anakin.bin";

// alignment of face
//std::string model_path = "/home/chaowen/anakin_v2/model_v2/anakin-models/adu/anakin_models/diepsie_light_head/net_deploy_stageI.anakin.bin";

//std::string model_path = "/home/chaowen/anakin_v2/model_v2/anakin-models/adu/anakin_models/diepsie_light_head/net_deploy_stageII.anakin.bin";

// residual 7 patch of face
//std::string model_path = "/home/chaowen/anakin_v2/model_v2/anakin-models/adu/anakin_models/diepsie_light_head/residual_net_7patch_3hc.anakin.bin";

// resnet 50
//std::string model_path = "/home/cuichaowen/anakin2/anakin2/benchmark/CNN/mobilenet_v2.anakin.bin";

// vgg16
std::string model_path = "/home/cuichaowen/anakin2/anakin2/benchmark/CNN/models/vgg16.anakin.bin";

#ifdef USE_CUDA
#if 1
TEST(NetTest, net_execute_base_test) {
    Graph<NV, AK_FLOAT, Precision::FP32>* graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << model_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_path);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    // reshape the input_0 's shape for graph model
    //graph->Reshape("input_0", {1, 8, 640, 640});

    // register all tensor inside graph
    //graph->RegistAllOut();

    // register edge
    // graph->RegistOut("conv2_2/expand/scale", "relu2_2/expand");

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
	{ // inner scope
#ifdef USE_DIEPSE
    Net<NV, AK_FLOAT, Precision::FP32, OpRunType::SYNC> net_executer(*graph, true);
#else
    Net<NV, AK_FLOAT, Precision::FP32> net_executer(*graph, true);
#endif

    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<Target_H, AK_FLOAT> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = h_tensor_in.mutable_data();

    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }

    d_tensor_in_p->copy_from(h_tensor_in);

#ifdef USE_DIEPSE
    // for diepse model
    auto d_tensor_in_1_p = net_executer.get_in("input_1");
    Tensor4d<X86, AK_FLOAT> h_tensor_in_1;

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
    Tensor4d<X86, AK_FLOAT> h_tensor_in_2;

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

    int epoch = 1;
    // do inference
    Context<NV> ctx(0, 0, 0);
    saber::SaberTimer<NV> my_time;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";
	// warm up
	/*for(int i=0; i<10; i++) {
		net_executer.prediction();
	}*/

    my_time.start(ctx);


    //auto start = std::chrono::system_clock::now();
    for(int i=0; i<epoch; i++) {
		//DLOG(ERROR) << " epoch(" << i << "/" << epoch << ") ";
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
    //LOG(WARNING) << "avg time : " << time/epoch <<" ms";

    my_time.end(ctx);
    LOG(INFO)<<"aveage time "<<my_time.get_average_ms()/epoch << " ms";

	} // inner scope over

	LOG(ERROR) << "inner net exe over !";

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
    // get out result
    //test_print<NV>(tensor_out_4_p);


    // save the optimized model to disk.
    /*std::string save_model_path = model_path + std::string(".saved");
    status = graph->save(save_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }*/
}
#endif 
#endif

#if 0
TEST(NetTest, net_execute_reconstruction_test) {
    graph = new Graph<NV, AK_FLOAT, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from optimized model " << model_saved_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_saved_path);
    if (!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    // regisiter output tensor
    //graph->RegistOut("data_perm",  "data_scale");
    graph->RegistOut("data_perm",  "conv1");

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
    Net<NV, AK_FLOAT, Precision::FP32> net_executer(*graph);

    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<X86, AK_FLOAT> h_tensor_in;

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
    Context<NV> ctx(0, 0, 0);
    saber::SaberTimer<NV> my_time;
    my_time.start(ctx);

    LOG(WARNING) << "EXECUTER !!!!!!!! ";
    for (int i=0; i<1; i++) {
        net_executer.prediction();

    }
    my_time.end(ctx);
    LOG(INFO)<<"aveage time "<<my_time.get_average_ms()/1 << " ms";

    auto tensor_out_inner_p = net_executer.get_tensor_from_edge("data_perm",  "conv1");

    // get out
    auto tensor_out_0_p = net_executer.get_out("loc_pred_out");
    auto tensor_out_1_p = net_executer.get_out("obj_pred_out");
    auto tensor_out_2_p = net_executer.get_out("cls_pred_out");
    auto tensor_out_3_p = net_executer.get_out("ori_pred_out");
    auto tensor_out_4_p = net_executer.get_out("dim_pred_out");

    
    // get out result
    test_print<NV>(tensor_out_inner_p);
}
#endif

int main(int argc, const char** argv){

	Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
