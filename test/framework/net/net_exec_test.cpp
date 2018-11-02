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
#elif defined(AMD_GPU)
using Target = AMD;
using Target_H = X86;
#endif

//#define USE_DIEPSE

std::string model_path = "/home/shixiaowei02/vgg16/vgg16.anakin.bin";

std::string model_saved_path = model_path + ".saved";

#ifdef AMD_GPU
#if 1
TEST(NetTest, net_execute_base_test) {
    Graph<AMD, Precision::FP32>* graph = new Graph<AMD, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << model_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_path);

    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    // reshape the input_0 's shape for graph model
    //graph->Reshape("input_0", {1, 8, 640, 640});
    //graph->ResetBatchSize("input_0", 2);

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
    //Net<AMD, Precision::FP32, OpRunType::SYNC> net_executer(*graph, true);
    Net<AMD, Precision::FP32, OpRunType::SYNC> net_executer(true);
#else
    //Net<AMD, Precision::FP32> net_executer(*graph, true);
    Net<AMD, Precision::FP32> net_executer(true);
#endif

   // net_executer.load_calibrator_config("net_pt_config.txt", "cal_file");
    net_executer.init(*graph);
    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<Target_H> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();

    for (int i = 0; i < valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = (float*)(h_tensor_in.mutable_data());

    for (int i = 0; i < h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }

    d_tensor_in_p->copy_from(h_tensor_in);


    int epoch = 1;
    // do inference
    Context<AMD> ctx(0, 0, 0);
    saber::SaberTimer<AMD> my_time;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";

    my_time.start(ctx);


    //auto sta/rt = std::chrono::system_clock::now();
    for (int i = 0; i < epoch; i++) {
        //DLOG(ERROR) << " epoch(" << i << "/" << epoch << ") ";
        net_executer.prediction();
    }

#ifdef ENABLE_OP_TIMER
    net_executer.print_and_reset_optime_summary();
#endif
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
    LOG(INFO) << "aveage time " << my_time.get_average_ms() / epoch << " ms";

    //} // inner scope over

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
    std::string save_model_path = model_path + std::string(".saved");
    status = graph->save(save_model_path);

    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    if (!graph) {
        delete graph;
    }
}
#endif
#endif

#if 0
TEST(NetTest, net_execute_reconstruction_test) {
    Graph<AMD, Precision::FP32>* graph = new Graph<AMD, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from optimized model " << model_saved_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_saved_path);

    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    // regisiter output tensor
    //graph->RegistOut("data_perm",  "data_scale");
    //graph->RegistOut("data_perm",  "conv1");

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
    Net<AMD, Precision::FP32> net_executer(*graph);

    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<X86> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();

    for (int i = 0; i < valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = h_tensor_in.mutable_data();

    for (int i = 0; i < h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }

    d_tensor_in_p->copy_from(h_tensor_in);

    // do inference
    Context<AMD> ctx(0, 0, 0);
    saber::SaberTimer<AMD> my_time;
    my_time.start(ctx);

    LOG(WARNING) << "EXECUTER !!!!!!!! ";

    for (int i = 0; i < 1; i++) {
        net_executer.prediction();

    }

    my_time.end(ctx);
    LOG(INFO) << "aveage time " << my_time.get_average_ms() / 1 << " ms";

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

int main(int argc, const char** argv) {
    if (argc < 2){
        LOG(INFO)<<"no input";
        return 0;
    }
    LOG(INFO)<<"begin";
    model_path = std::string(argv[1]);
    Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
