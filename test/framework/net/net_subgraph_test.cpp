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


TEST(NetTest, net_execute_subgraph_0) {
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>();

    std::vector<std::string> input{"x"};
    std::vector<std::string> output{"y"};

	graph->AddOp("op1", "Dense", input, output);
    graph->AddOpAttr("op1", "out_dim", 2);
    graph->AddOpAttr("op1", "bias_term", false);
    graph->AddOpAttr("op1", "axis", 3);
    std::vector<int> shape = {1, 1, 3, 2};
    anakin::saber::Shape tmp_shape{shape};
    PBlock<Target> weight1(tmp_shape);
    float *cpu_data = static_cast<float *>(weight1.h_tensor().mutable_data());
    for (int i = 0; i < 2 * 3; i++) { cpu_data[i] = i + 1; }

    weight1.d_tensor().set_shape(tmp_shape);
    weight1.d_tensor().copy_from(weight1.h_tensor());

    graph->AddOpAttr("op1", "weight_1", weight1);

    graph->Freeze();

    for (auto in : graph->get_ins()) {
    	LOG(INFO) << "get in: " <<  in;
    }

    for (auto out : graph->get_outs()) {
    	LOG(INFO) << "get out: " <<  out;
    }

    //anakin graph optimization
    graph->Optimize();

    anakin::PTuple<int> input_shape = {1, 1, 1, 3};
    graph->AddOpAttr("x", "input_shape", input_shape);

    //Net<Target, Precision::FP32> net_executer(true);
    std::unique_ptr<Net<Target, Precision::FP32> > net_executer_p(new Net<Target, Precision::FP32>(true));


    net_executer_p->init(*graph);

    auto d_tensor_in_p = net_executer_p->get_in("x");
    Tensor4d<Target_H> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect x dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = (float*)(h_tensor_in.mutable_data());

    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }
    d_tensor_in_p->copy_from(h_tensor_in);

    net_executer_p->prediction();

    auto tensor_out = net_executer_p->get_out("y");
    LOG(INFO) << "get output tensor:";
	test_print(tensor_out);
}

TEST(NetTest, net_execute_subgraph_three_fc_with_split) {
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>();

    auto add_fc_op = [&](const std::string& fc_name, 
                         const std::vector<std::string>& input, 
                         const std::vector<std::string>& output) {
        graph->AddOp(fc_name, "Dense", input, output);
        graph->AddOpAttr(fc_name, "out_dim", 5);
        graph->AddOpAttr(fc_name, "bias_term", false);
        graph->AddOpAttr(fc_name, "axis", 1);
        std::vector<int> shape = {1, 1, 5, 5};
        anakin::saber::Shape tmp_shape{shape};
        PBlock<Target> weight1(tmp_shape);
        float *cpu_data = static_cast<float *>(weight1.h_tensor().mutable_data());
        for (int i = 0; i < 5*5; i++) { cpu_data[i] = i + 1; }

        weight1.d_tensor().set_shape(tmp_shape);
        weight1.d_tensor().copy_from(weight1.h_tensor());

        graph->AddOpAttr(fc_name, "weight_1", weight1);

    };

    add_fc_op("op1", {"op1_in"}, {"temp"});
    add_fc_op("op2", {"temp"}, {"op2_out"});
    add_fc_op("op3", {"temp"}, {"op3_out"});

    auto status = graph->Freeze();
    if (!status){
        LOG(FATAL) << "Freeze error";
    }

    for (auto in : graph->get_ins()) {
    	LOG(INFO) << "get in: " <<  in;
    }

    for (auto out : graph->get_outs()) {
    	LOG(INFO) << "get out: " <<  out;
    }

    //anakin graph optimization
    graph->Optimize();

    // save the optimized model to disk.
    std::string save_model_path = std::string("subgraph.saved");
    status = graph->save(save_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }


    anakin::PTuple<int> input_shape = {1, 5, 1, 1};
    graph->AddOpAttr("op1_in", "input_shape", input_shape);

    //Net<Target, Precision::FP32> net_executer(true);
    std::unique_ptr<Net<Target, Precision::FP32> > net_executer_p(new Net<Target, Precision::FP32>(true));


    net_executer_p->init(*graph);

    auto d_tensor_in_p = net_executer_p->get_in("op1_in");
    Tensor4d<Target_H> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect x dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = (float*)(h_tensor_in.mutable_data());

    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }
    d_tensor_in_p->copy_from(h_tensor_in);

    net_executer_p->prediction();

    auto tensor_out_2 = net_executer_p->get_out("op2_out");
    LOG(INFO) << "get output tensor 2:";
	test_print(tensor_out_2);
    auto tensor_out_3 = net_executer_p->get_out("op3_out");
    LOG(INFO) << "get output tensor 3:";
	test_print(tensor_out_3);


}

TEST(NetTest, net_execute_subgraph_mult_fc) {
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>();

    auto add_fc_op = [&](const std::string& fc_name, 
                         const std::vector<std::string>& input, 
                         const std::vector<std::string>& output) {
        graph->AddOp(fc_name, "Dense", input, output);
        graph->AddOpAttr(fc_name, "out_dim", 1);
        graph->AddOpAttr(fc_name, "bias_term", false);
        graph->AddOpAttr(fc_name, "axis", 1);
        std::vector<int> shape = {1, 1, 1, 1};
        anakin::saber::Shape tmp_shape{shape};
        PBlock<Target> weight1(tmp_shape);
        float *cpu_data = static_cast<float *>(weight1.h_tensor().mutable_data());
        for (int i = 0; i < 1*1; i++) { cpu_data[i] = i + 1; }

        weight1.d_tensor().set_shape(tmp_shape);
        weight1.d_tensor().copy_from(weight1.h_tensor());

        graph->AddOpAttr(fc_name, "weight_1", weight1);

    };
    auto add_concat_op = [&](const std::string& cc_name, 
                             const std::vector<std::string>& input, 
                             const std::vector<std::string>& output) {
        graph->AddOp(cc_name, "concat", input, output);
        graph->AddOpAttr(cc_name, "axis", 3);
    };

    auto add_relu_op = [&](const std::string& relu_name, 
                           const std::vector<std::string>& input,
                           const std::vector<std::string>& output){ 
        graph->AddOp(relu_name, "ReLU", input, output); 
        graph->AddOpAttr(relu_name, "alpha", 0.0f);
    };



    add_fc_op("op0", {"x"}, {"out0"});
    add_fc_op("op1", {"x"}, {"out1"});
    add_fc_op("op2", {"x"}, {"out2"});
    add_fc_op("op3", {"x"}, {"out3"});
    add_fc_op("op4", {"x"}, {"out4"});
    add_fc_op("op5", {"x"}, {"out5"});
    add_fc_op("op6", {"x"}, {"out6"});
    add_concat_op("concat", {"out0", "out1", "out2", "out3", "out4", "out5", "out6"}, {"out_concat"});
    add_relu_op("relu", {"out_concat"}, {"out"});


	// this api should be called before freeze
	graph->RegistVar("out0");

    auto status = graph->Freeze();
    if (!status){
        LOG(FATAL) << "Freeze error";
    }

    for (auto in : graph->get_ins()) {
    	LOG(INFO) << "get in: " <<  in;
    }

    for (auto out : graph->get_outs()) {
    	LOG(INFO) << "get out: " <<  out;
    }

    //anakin graph optimization
    graph->Optimize();

    // save the optimized model to disk.
    std::string save_model_path = std::string("multi_fc_subgraph_with_regist_input.saved2");
    status = graph->save(save_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }


    anakin::PTuple<int> input_shape = {1, 1, 1, 1};
    graph->AddOpAttr("x", "input_shape", input_shape);

    //Net<Target, Precision::FP32> net_executer(true);
    std::unique_ptr<Net<Target, Precision::FP32> > net_executer_p(new Net<Target, Precision::FP32>(true));


    net_executer_p->init(*graph);

    auto d_tensor_in_p = net_executer_p->get_in("x");
    Tensor4d<Target_H> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect x dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = (float*)(h_tensor_in.mutable_data());

    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }
    d_tensor_in_p->copy_from(h_tensor_in);

    net_executer_p->prediction();

    //auto tensor_out = net_executer_p->get_out("out");
    //LOG(INFO) << "get output tensor";
	//test_print(tensor_out);
}

TEST(NetTest, net_execute_subgraph_concat) {
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>();

    auto add_concat_op = [&](const std::string& cc_name, 
                             const std::vector<std::string>& input, 
                             const std::vector<std::string>& output) {
        graph->AddOp(cc_name, "concat", input, output);
        graph->AddOpAttr(cc_name, "axis", 3);
    };

    add_concat_op("concat_1", {"x", "y"}, {"out"});

    auto status = graph->Freeze();
    if (!status){
        LOG(FATAL) << "Freeze error";
    }

    for (auto in : graph->get_ins()) {
    	LOG(INFO) << "get in: " <<  in;
    }

    for (auto out : graph->get_outs()) {
    	LOG(INFO) << "get out: " <<  out;
    }

    //anakin graph optimization
    graph->Optimize();

    // save the optimized model to disk.
    std::string save_model_path = std::string("concat_subgraph.saved2");
    status = graph->save(save_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }


    anakin::PTuple<int> input_shape_x = {1, 1, 5, 1};
    graph->AddOpAttr("x", "input_shape", input_shape_x);
    anakin::PTuple<int> input_shape_y = {1, 1, 5, 3};
    graph->AddOpAttr("y", "input_shape", input_shape_y);


    //Net<Target, Precision::FP32> net_executer(true);
    std::unique_ptr<Net<Target, Precision::FP32> > net_executer_p(new Net<Target, Precision::FP32>(true));


    net_executer_p->init(*graph);

    auto xd_tensor_in_p = net_executer_p->get_in("x");
    auto yd_tensor_in_p = net_executer_p->get_in("y");
    auto fill_tensor = [&](Tensor4d<Target> * d_tensor_p, float val) {
        Tensor4d<Target_H> h_tensor_in;

        auto valid_shape_in = d_tensor_p->valid_shape();
        for (int i=0; i<valid_shape_in.size(); i++) {
            LOG(INFO) << "detect fill tensor dims[" << i << "]" << valid_shape_in[i];
        }

        h_tensor_in.re_alloc(valid_shape_in);
        float* h_data = (float*)(h_tensor_in.mutable_data());

        for (int i=0; i<h_tensor_in.size(); i++) {
            h_data[i] = val;
        }
        d_tensor_p->copy_from(h_tensor_in);
    };

    fill_tensor(xd_tensor_in_p, 1.0);
    fill_tensor(yd_tensor_in_p, 2.0);


    net_executer_p->prediction();

    auto tensor_out = net_executer_p->get_out("out");
    LOG(INFO) << "get output tensor";
	test_print(tensor_out);
}

TEST(NetTest, net_execute_subgraph_eltwise) {
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>();

    auto add_eltwise_op = [&](const std::string& eltwise_name, 
                             const std::vector<std::string>& input, 
                             const std::vector<std::string>& output) {
        graph->AddOp(eltwise_name, "Eltwise", input, output);
        graph->AddOpAttr(eltwise_name, "type", std::string("Add"));
        anakin::PTuple<float> coeff;
        coeff.push_back(1.0);
        coeff.push_back(-1.0);
        LOG(INFO) << "coeff[0] " << coeff[0];
        //LOG(INFO) << "coeff[1] " << coeff[1];
        graph->AddOpAttr(eltwise_name, "coeff", coeff);

    };

    add_eltwise_op("eltwise", {"x", "y"}, {"out"});

    auto status = graph->Freeze();
    if (!status){
        LOG(FATAL) << "Freeze error";
    }

    for (auto in : graph->get_ins()) {
    	LOG(INFO) << "get in: " <<  in;
    }

    for (auto out : graph->get_outs()) {
    	LOG(INFO) << "get out: " <<  out;
    }

    //anakin graph optimization
    graph->Optimize();

    // save the optimized model to disk.
    std::string save_model_path = std::string("eltwise_subgraph.saved2");
    status = graph->save(save_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }


    anakin::PTuple<int> input_shape_x = {1, 1, 1, 3};
    graph->AddOpAttr("x", "input_shape", input_shape_x);
    anakin::PTuple<int> input_shape_y = {1, 1, 1, 3};
    graph->AddOpAttr("y", "input_shape", input_shape_y);


    //Net<Target, Precision::FP32> net_executer(true);
    std::unique_ptr<Net<Target, Precision::FP32> > net_executer_p(new Net<Target, Precision::FP32>(true));


    net_executer_p->init(*graph);

    auto xd_tensor_in_p = net_executer_p->get_in("x");
    auto yd_tensor_in_p = net_executer_p->get_in("y");
    auto fill_tensor = [&](Tensor4d<Target> * d_tensor_p, float val) {
        Tensor4d<Target_H> h_tensor_in;

        auto valid_shape_in = d_tensor_p->valid_shape();
        for (int i=0; i<valid_shape_in.size(); i++) {
            LOG(INFO) << "detect fill tensor dims[" << i << "]" << valid_shape_in[i];
        }

        h_tensor_in.re_alloc(valid_shape_in);
        float* h_data = (float*)(h_tensor_in.mutable_data());

        for (int i=0; i<h_tensor_in.size(); i++) {
            h_data[i] = val;
        }
        d_tensor_p->copy_from(h_tensor_in);
    };

    fill_tensor(xd_tensor_in_p, 2.0);
    fill_tensor(yd_tensor_in_p, 3.0);


    net_executer_p->prediction();

    auto tensor_out = net_executer_p->get_out("out");
    LOG(INFO) << "get output tensor";
	test_print(tensor_out);
}

TEST(NetTest, net_execute_subgraph_resnet_base_arch) {
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>();

    auto add_conv_op = [&](const std::string& conv_name, 
                         const std::vector<std::string>& input, 
                         const std::vector<std::string>& output) {
        graph->AddOp(conv_name, "Convolution", input, output);
        graph->AddOpAttr(conv_name, "group", 1);
        graph->AddOpAttr(conv_name, "bias_term", false);
        graph->AddOpAttr<PTuple<int>>(conv_name, "padding", {0, 0});
        graph->AddOpAttr<PTuple<int>>(conv_name, "strides", {1, 1});
        graph->AddOpAttr<PTuple<int>>(conv_name, "dilation_rate", {0, 0});
        graph->AddOpAttr(conv_name, "filter_num", 1);
        graph->AddOpAttr<PTuple<int>>(conv_name, "kernel_size", {1, 1});
        graph->AddOpAttr(conv_name, "axis", 1);

        std::vector<int> shape = {1, 1, 1, 1};
        anakin::saber::Shape tmp_shape{shape};
        auto* weight1 = graph::GraphGlobalMem<Target>::Global().template new_block<AK_FLOAT>(tmp_shape); 
        float *cpu_data = static_cast<float *>(weight1->h_tensor().mutable_data());
        for (int i = 0; i < 1*1; i++) { cpu_data[i] = i + 1; }

        weight1->d_tensor().set_shape(tmp_shape);
        weight1->d_tensor().copy_from(weight1->h_tensor());

        graph->AddOpAttr(conv_name, "weight_1", *weight1);

    };
    auto add_relu_op = [&](const std::string& relu_name, 
                           const std::vector<std::string>& input,
                           const std::vector<std::string>& output){ 
        graph->AddOp(relu_name, "ReLU", input, output); 
        graph->AddOpAttr(relu_name, "alpha", 0.0f);
    };

    auto add_eltwise_op = [&](const std::string& eltwise_name, 
                             const std::vector<std::string>& input, 
                             const std::vector<std::string>& output) {
        graph->AddOp(eltwise_name, "Eltwise", input, output);
        graph->AddOpAttr(eltwise_name, "type", std::string("Add"));
        anakin::PTuple<float> coeff;
        coeff.push_back(1.0);
        coeff.push_back(1.0);
        graph->AddOpAttr(eltwise_name, "coeff", coeff);

    };

    add_conv_op("conv_0", {"x"}, {"conv_0_out"});
    add_relu_op("conv_0_relu", {"conv_0_out"}, {"conv_0_relu_out"});
    add_conv_op("conv_1", {"conv_0_relu_out"}, {"conv_1_out"});
    add_eltwise_op("eltwise", {"conv_1_out", "conv_0_relu_out"}, {"out"});

    auto status = graph->Freeze();
    if (!status){
        LOG(FATAL) << "Freeze error";
    }

    for (auto in : graph->get_ins()) {
    	LOG(INFO) << "get in: " <<  in;
    }

    for (auto out : graph->get_outs()) {
    	LOG(INFO) << "get out: " <<  out;
    }

    //anakin graph optimization
    graph->Optimize();

    // save the optimized model to disk.
    std::string save_model_path = std::string("resnet_subgraph.saved2");
    status = graph->save(save_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }


    anakin::PTuple<int> input_shape_x = {1, 1, 1, 1};
    graph->AddOpAttr("x", "input_shape", input_shape_x);


    //Net<Target, Precision::FP32> net_executer(true);
    std::unique_ptr<Net<Target, Precision::FP32> > net_executer_p(new Net<Target, Precision::FP32>(true));


    net_executer_p->init(*graph);

    auto xd_tensor_in_p = net_executer_p->get_in("x");
    auto fill_tensor = [&](Tensor4d<Target> * d_tensor_p, float val) {
        Tensor4d<Target_H> h_tensor_in;

        auto valid_shape_in = d_tensor_p->valid_shape();
        for (int i=0; i<valid_shape_in.size(); i++) {
            LOG(INFO) << "detect fill tensor dims[" << i << "]" << valid_shape_in[i];
        }

        h_tensor_in.re_alloc(valid_shape_in);
        float* h_data = (float*)(h_tensor_in.mutable_data());

        for (int i=0; i<h_tensor_in.size(); i++) {
            h_data[i] = val;
        }
        d_tensor_p->copy_from(h_tensor_in);
    };

    fill_tensor(xd_tensor_in_p, 1.0);


    net_executer_p->prediction();

    auto tensor_out = net_executer_p->get_out("out");
    LOG(INFO) << "get output tensor";
	test_print(tensor_out);
}

TEST(NetTest, net_execute_subgraph_test_share_from) {
    // construct base gpu tensor
    std::vector<int> shape = {1, 1, 1, 5};
    anakin::saber::Shape tmp_shape{shape};
    Tensor4d<Target> d_tensor(tmp_shape);

    auto fill_tensor = [&](Tensor4d<Target> * d_tensor_p, float val) {
        Tensor4d<Target_H> h_tensor_in;

        auto valid_shape_in = d_tensor_p->valid_shape();
        for (int i=0; i<valid_shape_in.size(); i++) {
            LOG(INFO) << "detect fill tensor dims[" << i << "]" << valid_shape_in[i];
        }

        h_tensor_in.re_alloc(valid_shape_in);
        float* h_data = (float*)(h_tensor_in.mutable_data());

        for (int i=0; i<h_tensor_in.size(); i++) {
            h_data[i] = val;
        }
        d_tensor_p->copy_from(h_tensor_in);
    };

    fill_tensor(&d_tensor, 42.0f);


    Tensor4d<Target> shallow_d_tensor;//(tmp_shape);
    shallow_d_tensor.reshape(tmp_shape);
    {
        // construct shallow copy gpu tensor
        //Context<Target> ctx(0, 0, 0);
        //saber::SaberTimer<Target> my_time;
        //my_time.start(ctx);
        Tensor4d<Target> temp_tensor(d_tensor.mutable_data(), Target(), 0, tmp_shape); 
        //my_time.end(ctx);
        //LOG(INFO)<<"aveage time "<<my_time.get_average_ms() << " ms";

        shallow_d_tensor.share_from(temp_tensor);
    }
    // print all value
    auto p = &shallow_d_tensor;
    test_print(p);
}


int main(int argc, const char** argv){
	Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
