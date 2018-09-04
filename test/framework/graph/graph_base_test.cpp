#include <string>
#include "graph_test.h"
#include "graph_base.h"

using namespace anakin;
using namespace anakin::graph;

//! Usage sample
class GraphTestClass : public GraphBase<std::string, int, int> {
public:
    GraphTestClass() {}
    ~GraphTestClass() {}
    virtual bool directed() {
        return true;
    };
};
class edge : public Arc<std::string, int> {
public:
    edge(std::string btm, std::string top, int weight): Arc<std::string, int>(btm, top, weight) {}
    ~edge() {}
};

TEST(GraphTest, graph_base_test) {
    LOG(INFO) << "test for graph base .";

    GraphTestClass graph;
    graph.add_vertex("a", 42);
    graph.add_vertex("b", 43);
    graph.add_vertex("c", 44);
    graph.add_vertex("d", 45);
    graph.add_vertex("e", 46);
    graph.add_vertex("f", 47);

    edge arc0("a", "b", 0);
    edge arc1("b", "c", 1);
    edge arc2("c", "d", 2);
    edge arc3("d", "e", 3);
    edge arc4("e", "f", 4);
    edge arc5("f", "a", 5);

    graph.add_in_arc(arc0);
    graph.add_in_arc(arc1);
    graph.add_in_arc(arc2);
    graph.add_in_arc(arc3);
    graph.add_in_arc(arc4);
    graph.add_in_arc(arc5);
    graph.add_out_arc(arc0);
    graph.add_out_arc(arc1);
    graph.add_out_arc(arc2);
    graph.add_out_arc(arc3);
    graph.add_out_arc(arc4);
    graph.add_out_arc(arc5);

    LOG(WARNING) << "Construction of graph.";
    LOG(INFO) << graph.to_string();

    LOG(WARNING) << "Remove a from graph.";
    graph.remove("a");
    LOG(INFO) << graph.to_string();

    LOG(WARNING) << "Add arc: f->b to graph.";
    edge arc_f_b("f", "b", 10);
    graph.add_in_arc(arc_f_b);
    graph.add_out_arc(arc_f_b);
    LOG(INFO) << graph.to_string();

    LOG(WARNING) << "Add vertex:a and arc: a->e to graph.";
    graph.add_vertex("a", 47);
    edge arc_a_e("a", "e", 10);
    graph.add_out_arc(arc_a_e);
    graph.add_in_arc(arc_a_e);
    LOG(INFO) << graph.to_string();
}


int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
