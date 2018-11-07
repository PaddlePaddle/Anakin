#include <string>
#include "graph_test.h"
#include "graph_base.h"
#include "framework/graph/graph.h"
#include "framework/graph/llvm/virtual_graph.h"
#include "framework/graph/node.h"
#include "framework/graph/llvm/fusion/graph_pattern.h"
#include "framework/graph/llvm/scheduler.h"
#include "framework/core/net/net.h"
#include "saber/core/tensor_op.h"

using namespace anakin;
using namespace anakin::graph;

//! Usage sample

class edge : public Arc<std::string, io> {
public:
    edge(std::string btm, std::string top, io weight): Arc<std::string, io>(btm, top, weight) {}
    ~edge() {}
};

void test_inception() {
#ifdef USE_CUDA
    //mode path
    std::string model_path = "/home/tianxiaogang/txg/ps/ps_origin.anakin.bin";
    Graph<NV, Precision::FP32> graph;
    //load
    LOG(INFO) << "loading the model";
    auto status = graph.load(model_path);

    if (!status) {
        LOG(FATAL) << "load model error!!" << status.info();
    }

    //optimize
    LOG(INFO) << "optimizing the model";
    graph.Optimize();
    //save
    std::string save_model_path = "optimized_inception.save";
    LOG(INFO) << "saving model";
    status = graph.save(save_model_path);

    if (!status) {
        LOG(FATAL) << "save model error!!" << status.info();
    }

    //execute
    LOG(INFO) << "create net to execute";
    Net<NV, Precision::FP32> net_executer(graph, true);
    //get inputs
    LOG(INFO) << "get input";
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<X86> h_tensor_in;
    auto valid_shape_in = d_tensor_in_p->valid_shape();

    for (int i = 0; i < valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    fill_tensor_rand(h_tensor_in, -1.0f, 1.0f);
    d_tensor_in_p->copy_from(h_tensor_in);

    //do inference
    Context<NV> ctx(0, 0, 0);
    LOG(WARNING) << "EXECUTER !!!!!!!! ";

    for (int i = 0; i < 2; i++) {
        net_executer.prediction();
    }

#endif
}

void test_virtual() {
    VGraph graph;
    node n1;
    n1.name = "1";
    n1.opName = "input";
    graph.add_vertex("1", n1);
    node n2;
    n2.name = "2";
    n2.opName = "conv";
    graph.add_vertex("2", n2);
    node n3;
    n3.name = "3";
    n3.opName = "split";
    graph.add_vertex("3", n3);
    node n4;
    n4.name = "4";
    n4.opName = "embedding";
    graph.add_vertex("4", n4);
    node n5;
    n5.name = "5";
    n5.opName = "embedding";
    graph.add_vertex("5", n5);
    node n6;
    n6.name = "6";
    n6.opName = "embedding";
    graph.add_vertex("6", n6);
    node n7;
    n7.name = "7";
    n7.opName = "pooling";
    graph.add_vertex("7", n7);
    node n8;
    n8.name = "8";
    n8.opName = "embedding";
    graph.add_vertex("8", n8);
    node n9;
    n9.name = "9";
    n9.opName = "embedding";
    graph.add_vertex("9", n9);

    node n10;
    n10.name = "10";
    n10.opName = "embedding";
    graph.add_vertex("10", n10);

    node n11;
    n11.name = "11";
    n11.opName = "embedding";
    graph.add_vertex("11", n11);

    node n12;
    n12.name = "12";
    n12.opName = "concat";
    graph.add_vertex("12", n12);

    node n13;
    n13.name = "13";
    n13.opName = "embedding";
    graph.add_vertex("13", n13);
    node n14;
    n14.name = "14";
    n14.opName = "embedding";
    graph.add_vertex("14", n14);

    node n15;
    n15.name = "15";
    n15.opName = "embedding";
    graph.add_vertex("15", n15);

    node n16;
    n16.name = "16";
    n16.opName = "embedding";
    graph.add_vertex("16", n16);

    node n17;
    n17.name = "17";
    n17.opName = "pooling";
    graph.add_vertex("17", n17);

    node n18;
    n18.name = "18";
    n18.opName = "embedding";
    graph.add_vertex("18", n18);

    node n19;
    n19.name = "19";
    n19.opName = "embedding";
    graph.add_vertex("19", n19);

    node n20;
    n20.name = "20";
    n20.opName = "embedding";
    graph.add_vertex("20", n20);

    node n21;
    n21.name = "21";
    n21.opName = "concat";
    graph.add_vertex("21", n21);







    io new_io;
    edge arc0("1", "2", new_io);
    edge arc1("2", "3", new_io);
    edge arc2("3", "4", new_io);
    edge arc3("3", "5", new_io);
    edge arc4("3", "7", new_io);
    edge arc5("3", "9", new_io);
    edge arc6("4", "12", new_io);
    edge arc7("5", "6", new_io);
    edge arc8("6", "12", new_io);
    edge arc9("7", "8", new_io);
    edge arc10("8", "12", new_io);
    edge arc11("9", "10", new_io);
    edge arc12("10", "11", new_io);
    edge arc13("11", "12", new_io);
    edge arc14("12", "13", new_io);
    edge arc15("12", "14", new_io);
    edge arc16("12", "16", new_io);
    edge arc17("12", "18", new_io);
    edge arc18("13", "21", new_io);
    edge arc19("14", "15", new_io);
    edge arc20("15", "21", new_io);
    edge arc21("16", "17", new_io);
    edge arc22("17", "21", new_io);
    edge arc23("18", "19", new_io);
    edge arc24("19", "20", new_io);
    edge arc25("20", "21", new_io);

    graph.add_in_arc(arc0);
    graph.add_in_arc(arc1);
    graph.add_in_arc(arc2);
    graph.add_in_arc(arc3);
    graph.add_in_arc(arc4);
    graph.add_in_arc(arc5);
    graph.add_in_arc(arc6);
    graph.add_in_arc(arc7);
    graph.add_in_arc(arc8);
    graph.add_in_arc(arc9);
    graph.add_in_arc(arc10);
    graph.add_in_arc(arc11);
    graph.add_in_arc(arc12);
    graph.add_in_arc(arc13);
    graph.add_in_arc(arc14);
    graph.add_in_arc(arc15);
    graph.add_in_arc(arc16);
    graph.add_in_arc(arc17);
    graph.add_in_arc(arc18);
    graph.add_in_arc(arc19);
    graph.add_in_arc(arc20);
    graph.add_in_arc(arc21);
    graph.add_in_arc(arc22);
    graph.add_in_arc(arc23);
    graph.add_in_arc(arc24);
    graph.add_in_arc(arc25);



    graph.add_out_arc(arc0);
    graph.add_out_arc(arc1);
    graph.add_out_arc(arc2);
    graph.add_out_arc(arc3);
    graph.add_out_arc(arc4);
    graph.add_out_arc(arc5);
    graph.add_out_arc(arc6);
    graph.add_out_arc(arc7);
    graph.add_out_arc(arc8);
    graph.add_out_arc(arc9);
    graph.add_out_arc(arc10);
    graph.add_out_arc(arc11);
    graph.add_out_arc(arc12);
    graph.add_out_arc(arc13);
    graph.add_out_arc(arc14);
    graph.add_out_arc(arc15);
    graph.add_out_arc(arc16);
    graph.add_out_arc(arc17);
    graph.add_out_arc(arc18);
    graph.add_out_arc(arc19);
    graph.add_out_arc(arc20);
    graph.add_out_arc(arc21);
    graph.add_out_arc(arc22);
    graph.add_out_arc(arc23);
    graph.add_out_arc(arc24);
    graph.add_out_arc(arc25);

    auto in_parellel_fusion_op_name_vec =
        FusionOpRegister::Global().get_list_op_name_in_fusion_order_of(IN_PARELLEL);

    for (auto & fusion_name : in_parellel_fusion_op_name_vec) {
        LOG(INFO) << " processing in-parellel fusion : " << fusion_name;
        graph.Match(FusionOpRegister::Global()[fusion_name]);

        for (auto it = graph.begin(); it != graph.end(); ++it) {
            printf("node:\n");
            printf("first name:%s, ", it->first.c_str());
            printf("node.name:%s", it->second.name.c_str());
            printf("\n");
        }

        DLOG(INFO) << graph.to_string();
    }

    LOG(INFO) << "scheduler...";
    Scheduler scheduler;
    scheduler.RegIOResource(&graph);
    scheduler.Run();
}


TEST(GraphTest, graph_base_test) {
    LOG(INFO) << "test virtual::";
    //test_virtual();
    LOG(INFO) << "test inception:";
    test_inception();

}


int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
