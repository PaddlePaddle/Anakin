#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include "saber/core/tensor_op.h"
#include <chrono>
#include "debug.h"

#ifdef ENABLE_OP_TIMER
#include "saber/funcs/impl/impl_base.h"
#endif

std::string g_model_path = "";
std::string g_bmodel_path = "";
int g_batch_size = 1;

#ifdef USE_BM_PLACE

TEST(NetTest, net_execute_base_test) {

    LOG(INFO) << "begin test";
    auto ctx_p = std::make_shared<Context<BM>>();
    ctx_p->set_bmodel_path(g_bmodel_path);
  
  
    Graph<BM, Precision::FP32>* graph = new Graph<BM, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";

    // load anakin model files.
    auto status = graph->load(g_model_path);
    if (!status) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
      
    std::vector<std::string>& vin_name = graph->get_ins();
    LOG(INFO) << "number of input tensor: " << vin_name.size();

    for (int j = 0; j < vin_name.size(); ++j) {
        graph->ResetBatchSize("input_0", g_batch_size);
    }

    graph->fusion_optimize();
    Net<BM, Precision::FP32> net_executer(true);
#if 1
    net_executer.fusion_init(*graph, ctx_p, true);
    auto in_tensor_list = net_executer.get_in_list();
    auto out_tensor_list = net_executer.get_out_list();

    Tensor<BM> h_tensor_in;
    for (int j = 0; j < vin_name.size(); ++j)
    {
        auto d_tensor_in_p = net_executer.get_in(vin_name[j]);
        LOG(INFO) << "input name " << vin_name[j];
        LOG(INFO) << "input tensor size: " << d_tensor_in_p->size();
        auto shape_in = d_tensor_in_p->valid_shape();
        for (int i = 0; i < shape_in.size(); i++) {
            LOG(INFO) << "detect input_0 dims[" << i << "]" << shape_in[i];
        }

    }
    
    net_executer.fusion_prediction();
#endif
    delete graph;
}
int main(int argc, const char** argv) {
    LOG(INFO)<< "usage:";
    LOG(INFO)<< argv[0] << " <anakin model> <bmodel> <num> ";
    LOG(INFO)<< "   anakin model:     path to anakin model";
    LOG(INFO)<< "   bmodel :     path to bmodel";
    LOG(INFO)<< "   num:            batchSize default to 1";
  
    g_model_path = std::string(argv[1]);
    g_bmodel_path = std::string(argv[2]);

    if (argc > 3) {
        g_batch_size = atoi(argv[3]);
    }

    Env<BM>::env_init();
    
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);

    return 0;
}
#else
int main(int argc, const char** argv) {
    return 0;
}
#endif

