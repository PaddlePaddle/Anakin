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

std::string model_path = "/home/lxc/projects/models/converter_lego/output/ps.anakin.bin";

std::string model_saved_path = model_path + ".saved";

#ifdef USE_CUDA
#if 1
TEST(NetTest, net_execute_base_test) {
    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
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
    Net<NV, Precision::FP32, OpRunType::SYNC> net_executer(*graph, true);
#else
    Net<NV, Precision::FP32> net_executer(*graph, true);
#endif
    int batch_size = 1;
    int max_len = 256;

    // get in
    auto graph_ins = graph->get_ins();
    //  std::vector<int> offset_;
    //  std::vector<std::vector<int>> seq_offset;
    //  for(int i = 0; i < batch_size + 1; i++) {
    //      offset_.push_back(i*max_len);
    //  }
    //  seq_offset.push_back(offset_);
    //  for (auto input_name : graph_ins) {
    //      Shape input_shape = std::vector<int>{batch_size * max_len, 1 ,1, 1};
    //      auto d_tensor_in_p = net_executer.get_in(input_name);
    //      d_tensor_in_p->reshape(input_shape);
    //      Tensor4d<Target_H> h_tensor_in;
    //      h_tensor_in.reshape(input_shape);
    //      auto valid_shape_in = d_tensor_in_p->valid_shape();
    //      for (int i=0; i<valid_shape_in.size(); i++) {
    //          LOG(INFO) << "detect input_0:"<< input_name << "dims[" << i << "]" << valid_shape_in[i];
    //      }
    ////        h_tensor_in.re_alloc(valid_shape_in);
    //      float* h_data = (float*)(h_tensor_in.mutable_data());
    //
    //      for (int i=0; i<h_tensor_in.valid_size(); i++) {
    //          h_data[i] = 1.0f;
    //      }
    //      d_tensor_in_p->set_seq_offset(seq_offset);
    //      d_tensor_in_p->copy_from(h_tensor_in);
    //  }
    auto d_query_tensor_in_p = net_executer.get_in("query_basic_input");
    Tensor4d<Target_H> h_query_tensor_in;
    std::vector<int> offset_query;
    //    int query_len = std::rand() % max_len + 1;
    int query_len = 40;

    for (int i = 0; i < batch_size + 1; i++) {
        offset_query.push_back(query_len * i);
    }

    Shape  query_shape = std::vector<int> {query_len * batch_size, 1, 1, 1};
    d_query_tensor_in_p->reshape(query_shape);
    h_query_tensor_in.reshape(query_shape);
    // fill_tensor_rand(h_query_tensor_in);
    float* h_query = (float*)h_query_tensor_in.mutable_data();

    for (int i = 0; i < h_query_tensor_in.valid_size(); i++) {
        h_query[i] = 1.0f;
    }

    d_query_tensor_in_p->copy_from(h_query_tensor_in);
    std::vector<std::vector<int>>offset_query0 {offset_query};
    d_query_tensor_in_p->set_seq_offset(offset_query0);

    auto d_pos_tensor_in_p = net_executer.get_in("pos_abs_basic_input");
    Tensor4d<Target_H> h_pos_tensor_in;
    std::vector<int> offset_pos;
    //  int pos_accum_len = 0;
    //  offset_pos.push_back(pos_accum_len);
    //  for (int i = 0; i < batch_size; i++){
    //        int pos_len = std::rand() % max_len + 1;
    //      pos_accum_len += pos_len;
    //      offset_pos.push_back(pos_accum_len);
    //  }

    int pos_accum_len = 40;

    for (int i = 0; i < batch_size + 1; i++) {
        //       int pos_len = std::rand() % max_len + 1;
        //      pos_accum_len += pos_len;
        offset_pos.push_back(pos_accum_len * i);
    }

    Shape pos_shape = std::vector<int> {pos_accum_len * batch_size, 1, 1, 1};
    d_pos_tensor_in_p->reshape(pos_shape);
    h_pos_tensor_in.reshape(pos_shape);
    // fill_tensor_rand(h_pos_tensor_in);
    float* h_pos = (float*)h_pos_tensor_in.mutable_data();

    for (int i = 0; i < h_pos_tensor_in.valid_size(); i++) {
        h_pos[i] = 1.0f;
    }

    std::vector<std::vector<int>>offset_pos0 {offset_pos};
    d_pos_tensor_in_p->copy_from(h_pos_tensor_in);
    d_pos_tensor_in_p->set_seq_offset(offset_pos0);

    auto d_pos_title_tensor_in_p = net_executer.get_in("pos_title_basic_input");
    Tensor4d<Target_H> h_pos_title_tensor_in;
    std::vector<int> offset_pos_title;
    int pos_title_accum_len = 40;

    //  offset_pos_title.push_back(pos_title_accum_len);
    for (int i = 0; i < batch_size + 1; i++) {
        //      int pos_len = std::rand() % max_len + 1;
        //      pos_title_accum_len += pos_len;
        offset_pos_title.push_back(pos_title_accum_len * i);
    }

    Shape pos_title_shape = std::vector<int> {pos_title_accum_len * batch_size, 1, 1, 1};
    d_pos_title_tensor_in_p->reshape(pos_title_shape);
    h_pos_title_tensor_in.reshape(pos_title_shape);
    //  fill_tensor_rand(h_pos_title_tensor_in);
    float* h_pos_1 = (float*)h_pos_title_tensor_in.mutable_data();

    for (int i = 0; i < h_pos_title_tensor_in.valid_size(); i++) {
        h_pos_1[i] = 1.0f;
    }

    d_pos_title_tensor_in_p->copy_from(h_pos_title_tensor_in);
    std::vector<std::vector<int>>offset_pos_title0 {offset_pos_title};
    d_pos_title_tensor_in_p->set_seq_offset(offset_pos_title0);

    //#ifdef USE_DIEPSE
#if  0
    // for diepse model
    LOG(INFO) << "this is input_1";
    auto d_tensor_in_1_p = net_executer.get_in("input_1");
    Tensor4d<X86> h_tensor_in_1;

    h_tensor_in_1.re_alloc(d_tensor_in_1_p->valid_shape());

    for (int i = 0; i < d_tensor_in_1_p->valid_shape().size(); i++) {
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

    LOG(INFO) << "this is input_2";
    h_tensor_in_2.re_alloc(d_tensor_in_2_p->valid_shape());

    for (int i = 0; i < d_tensor_in_2_p->valid_shape().size(); i++) {
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
    for (int i = 0; i < epoch; i++) {
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
    auto tensor_out_0_p = net_executer.get_out("qps_out");


    // get out result
    //LOG(WARNING)<< "result avg: " << tensor_average(tensor_out_0_p);
    test_print(tensor_out_0_p);



    // save the optimized model to disk.
    std::string save_model_path = model_path + std::string(".saved");
    status = graph->save(save_model_path);

    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    delete graph;
}
#endif
#endif

#if 0
TEST(NetTest, net_execute_reconstruction_test) {
    Graph<NV, Precision::FP32>* graph = new Graph<NV, Precision::FP32>();
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
    Net<NV, Precision::FP32> net_executer(*graph);

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
    Context<NV> ctx(0, 0, 0);
    saber::SaberTimer<NV> my_time;
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

    Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
