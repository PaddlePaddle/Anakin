#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>

#ifdef USE_CUDA
using Target = NV;
using Target_H = X86;
#endif
#ifdef USE_X86_PLACE
using Target = X86;
using Target_H = X86;
#endif
#ifdef USE_ARM_PLACE
using Target = ARM;
using Target_H = ARM;
#endif


//std::string model_path = "/home/shixiaowei/temp/vgg19_trans.bin";
//std::string model_path = "/home/shixiaowei/temp/mobilenet_ssd/mobilenet_ssd.bin";
//std::string model_path = "/home/public/model_from_fluid/SE-ResNeXt-50/se-resnext50_fluid_anakin2.bin";
std::string model_path = "/home/liujunjie/macbuild/models/ocr_anakin_check/ocr20_var_05190007.bin";


//std::string model_path = "/home/shixiaowei/temp/vgg19_fluid/vgg19_fluid.anakin.bin";
//std::string model_path = "/home/public/anakin-models/public/vgg19_caffe.anakin.bin";

#if 1
TEST(NetTest, net_execute_base_test) {
    graph = new Graph<Target, AK_FLOAT, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << model_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_path);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    // reshape the input_0 's shape for graph model
    //graph->Reshape("input_0", {1, 8, 640, 640});
    graph->ResetBatchSize("input_0", 1);

    // register all tensor inside graph
    // graph->RegistAllOut();

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
    Net<Target, AK_FLOAT, Precision::FP32> net_executer(*graph, true);

    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<Target_H, AK_FLOAT> h_tensor_in;

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


    int epoch = 1;
    // do inference
    Context<Target> ctx(0, 0, 0);
    saber::SaberTimer<Target> my_time;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";
	// warm up
	for(int i=0; i<epoch; i++) {
		net_executer.prediction();
	}

    //auto& tensor_out_inner_p = net_executer.get_tensor_from_edge("data_perm", "conv1");

    //auto tensor_out_0_p = net_executer.get_out("detection_output_0.tmp_0_out");

    //auto tensor_out_0_p = net_executer.get_out("fc_32.tmp_2_out");  // resnext50_fluid
    //auto tensor_out_0_p = net_executer.get_out("fc_2.tmp_3_out");  // ocr20_var

    //auto tensor_out_0_p = net_executer.get_out("prob_out");  // vgg19_caffe
    //auto tensor_out_0_p = net_executer.get_out("softmax_0.tmp_0_out");  // vgg19_fluid

    //test_print(tensor_out_0_p);




    //auto tensor_out_1_p = net_executer.get_tensor_from_edge("mul__60", "softmax__62");  // vgg19_fluid
    //test_print(tensor_out_1_p);



    auto tensor_out_1_p = net_executer.get_tensor_from_edge("im2sequence__64", "__split_im2sequence__64__s0");  // ocr20_var
    //test_print(tensor_out_1_p);


    // save the optimized model to disk.
    /*std::string save_model_path = model_path + std::string(".saved");
    status = graph->save(save_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }*/
}
#endif 





int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
