#include "framework/model_parser/parser/parser.h"
#include "framework/model_parser/parser/model_io.h"
#include "graph.pb.h"
#include "node.pb.h"
#include "operator.pb.h"
#include "tensor.pb.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>

#ifdef USE_NANOPB
#include "nanopb/adapter.h"
#else
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/text_format.h>
#endif

namespace anakin {

namespace parser {

template<typename Ttype, DataType Dtype, Precision Ptype>
Status load(graph::Graph<Ttype, Dtype, Ptype>* graph, std::string& model_path) {
    return load(graph, model_path.c_str());
}

Status parse_graph_proto(GraphProto& graph_proto, const char* buffer, size_t len) {
#ifdef USE_NANOPB
    bool success = graph_proto.parse_from_buffer(buffer, len);
#else
    google::protobuf::io::ArrayInputStream raw_input(buffer, len);
    google::protobuf::io::CodedInputStream coded_input(&raw_input);
    coded_input.SetTotalBytesLimit(INT_MAX, 536870912);
    bool success = graph_proto.ParseFromCodedStream(&coded_input) && coded_input.ConsumedEntireMessage();
#endif
    if (!success) {
        LOG(FATAL) << " Parsing GraphProto " << " ERROR";
    }
    return Status::OK();
}

#ifndef USE_NANOPB
Status parse_graph_proto(GraphProto& graph_proto, std::istream* instream){
    if (!graph_proto.ParseFromIstream(instream)) {
        DLOG(ERROR) << "Fail to parse GraphProto.";
        return Status::FAIL("Fail to parse GraphProto.");
    }
    return Status::OK();
}
#endif

Status parse_graph_proto(GraphProto& graph_proto, const char* model_path) {
#ifdef USE_NANOPB
  FILE *f = fopen(model_path, "r");
  graph_proto.parse_from_file(f);
  fclose(f);
  return Status::OK();
#else

#if 0
    std::fstream input(model_path, std::ios::in | std::ios::binary);

    if (!input) {
        DLOG(ERROR) << model_path << " : File not found. ";
        return Status::FAIL("File not found");
    }

    GraphProto graph_proto;

    // parsing GraphProto from model
    if (!graph_proto.ParseFromIstream(&input)) {
        DLOG(ERROR) << "Fail to parse GraphProto.";
        return Status::FAIL("Fail to parse GraphProto.");
    }

#else
    int file_descriptor = open(model_path, O_RDONLY);

    if (file_descriptor == -1) {
        LOG(FATAL) << " Cant open " << model_path;
    }

    google::protobuf::io::FileInputStream raw_input(file_descriptor);

    google::protobuf::io::CodedInputStream coded_input(&raw_input);

    coded_input.SetTotalBytesLimit(ProtoReadBytesLimit, 536870912);

    bool success = graph_proto.ParseFromCodedStream(&coded_input);

    if (!success) {
        LOG(FATAL) << " Parsing GraphProto " << model_path << " ERROR";
    }

    close(file_descriptor);
    return Status::OK();
#endif
#endif
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status generate_graph_with_graph_proto(graph::Graph<Ttype, Dtype, Ptype>* graph, GraphProto& graph_proto) {
    // fill the graph with name
    LOG(INFO) << "graph name: " << graph_proto.name();
    graph->set_name(graph_proto.name());

    // fill the graph with ins/outs
    for (int i = 0; i < graph_proto.ins().size(); i++) {
        LOG(INFO) << "graph in: " << graph_proto.ins()[i];
        std::string in_name(graph_proto.ins()[i]);
        graph->add_in(in_name);
    }

    for (int i = 0; i < graph_proto.outs().size(); i++) {
        LOG(INFO) << "graph out: " << graph_proto.outs()[i];
        std::string out_name(graph_proto.outs()[i]);
        graph->add_out(out_name);
    }

    // fill the graph with nodes
    NodeIO<Ttype, Dtype, Ptype> node_io;

    for (int i = 0; i < graph_proto.nodes().size(); i++) {
        node_io >> graph_proto.nodes()[i];
    }

    node_io << *graph;

    // fill the graph with edges
    auto it_in = graph_proto.edges_in().begin();

    for (; it_in != graph_proto.edges_in().end(); ++it_in) {
        DLOG(WARNING) << " Parsing in edges of node : " << it_in->first;
        auto& key = it_in->first;
        auto& second = it_in->second;

        for (int i = 0; i < second.val().size(); i++) {
            //Tensor4dPtr<Ttype, Dtype> tensor_p = std::make_shared<Tensor4d<Ttype, Dtype>>();
            graph::Edge<Ttype, Dtype> edge(second.val()[i], key);
            //edge.weight() = new Tensor4d<Ttype, Dtype>();
            //edge.weight() = std::make_shared<Tensor4d<Ttype, Dtype> >();
            edge.shared() = (*graph_proto.mutable_edges_info())[edge.name()].shared();
            edge.share_from() = (*graph_proto.mutable_edges_info())[edge.name()].share_from();
            graph->add_in_arc(edge);
        }
    }

    auto it_out = graph_proto.edges_out().begin();

    for (; it_out != graph_proto.edges_out().end(); ++it_out) {
        auto& key = it_out->first;
        auto& second = it_out->second;

        for (int i = 0; i < second.val().size(); i++) {
            //Tensor4dPtr<Ttype, Dtype> tensor_p = std::make_shared<Tensor4d<Ttype, Dtype>>();
            graph::Edge<Ttype, Dtype> edge(key, second.val()[i]);
            //edge.weight() = new Tensor4d<Ttype, Dtype>();
            //edge.weight() = std::make_shared<Tensor4d<Ttype, Dtype> >();
            edge.shared() = (*graph_proto.mutable_edges_info())[edge.name()].shared();
            edge.share_from() = (*graph_proto.mutable_edges_info())[edge.name()].share_from();
            graph->add_out_arc(edge);
        }
    }


    // fill the graph with edges
    /*for(int i=0; i < node_io.get_node_name_in_order().size(); i++) {
        auto& node_name = node_io.get_node_name_in_order()[i];
        if (graph_proto.edges().count(node_name) > 0) {
            auto& second_node_name_list = graph_proto.edges().at(node_name);
            for(int j = 0; j < second_node_name_list.val().size(); j++) {
                graph::Edge<Ttype, Dtype> edge(node_name, second_node_name_list.val()[j]);
                edge.weight() = std::make_shared<Tensor4d<Ttype, Dtype> >();
                edge.shared() = (*graph_proto.mutable_edges_info())[edge.name()].shared();
                edge.share_from() = (*graph_proto.mutable_edges_info())[edge.name()].share_from();
                graph->add_arc(edge);
            }
        } else {
            LOG(FATAL) << " Node : " << node_name << " not found!";
        }
    }*/

    // fill the graph with info (only use the key value: is_optimized)
    graph->statistics.template set_info<graph::IS_OPTIMIZED>(graph_proto.summary().is_optimized());
    graph->statistics.template set_info<graph::TEMP_MEM>(graph_proto.summary().temp_mem_used());
    graph->statistics.template set_info<graph::ORI_TEMP_MEM>
    (graph_proto.summary().original_temp_mem_used());
    graph->statistics.template set_info<graph::SYSTEM_MEM>(graph_proto.summary().system_mem_used());
    graph->statistics.template set_info<graph::MODEL_MEM>(graph_proto.summary().model_mem_used());

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status load(graph::Graph<Ttype, Dtype, Ptype>* graph, const char* model_path) {
    GraphProto graph_proto;
    parse_graph_proto(graph_proto, model_path);
    return generate_graph_with_graph_proto(graph, graph_proto);
}

#ifndef USE_NANOPB
template<typename Ttype, DataType Dtype, Precision Ptype>
Status load(graph::Graph<Ttype, Dtype, Ptype>* graph, std::istream* instream) {

    GraphProto graph_proto;
    parse_graph_proto(graph_proto, instream);
    return generate_graph_with_graph_proto(graph, graph_proto);;
}
#endif

template<typename Ttype, DataType Dtype, Precision Ptype>
Status load(graph::Graph<Ttype, Dtype, Ptype>* graph, const char* buffer, size_t len) {

    GraphProto graph_proto;
    parse_graph_proto(graph_proto, buffer, len);
    return generate_graph_with_graph_proto(graph, graph_proto);;
}

#ifndef USE_NANOPB
template<typename Ttype, DataType Dtype, Precision Ptype>
Status save(graph::Graph<Ttype, Dtype, Ptype>* graph, std::string& model_path) {
    return save(graph, model_path.c_str());
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status save(graph::Graph<Ttype, Dtype, Ptype>* graph, const char* model_path) {
    std::fstream output(model_path, std::ios::out | std::ios::trunc | std::ios::binary);

    if (!output) {
        LOG(ERROR) << model_path << " : File not found. ";
        return Status::FAIL("File not found");
    }

    GraphProto graph_proto;
    // TODO...  fill the graph_proto with graph.
    // set graph proto name
    graph_proto.set_name(graph->name());

    // fill the graph proto with ins/outs
    for (auto in : graph->get_ins()) {
        graph_proto.add_ins(in);
    }

    for (auto out : graph->get_outs()) {
        graph_proto.add_outs(out);
    }

    // fill the graph proto  nodes with NodePtr in exec order
    NodeIO<Ttype, Dtype, Ptype> node_io;
    auto nodes_in_exec_order = graph->get_nodes_in_order();

    for (int i = 0; i < nodes_in_exec_order.size(); i++) {
        node_io >> (*graph)[nodes_in_exec_order[i]];;
    }

    node_io << graph_proto;

    // fill the graph proto' edges/edges_info with edges
    auto edges_in = graph_proto.mutable_edges_in();
    auto edges_out = graph_proto.mutable_edges_out();
    auto edges_info = graph_proto.mutable_edges_info();
    /*auto insert_edge = [&](graph::Edge<Ttype, Dtype>& edge) {
        (*edges)[edge.first()].add_val(edge.second());
        TensorProto ts;
        ts.set_name(edge.name());
        ts.set_shared(edge.shared());
        ts.set_share_from(edge.share_from());
        (*edges_info)[edge.name()].CopyFrom(ts);
    };*/
    auto insert_edge = [&](graph::NodePtr<Ttype, Dtype, Ptype>& node_p) {
        auto& arcs_it_in = graph->get_in_arc_its(node_p->name());
        auto& arcs_it_out = graph->get_out_arc_its(node_p->name());

        for (auto& edge_it : arcs_it_in) {
            (*edges_in)[edge_it->second()].add_val(edge_it->first());
            TensorProto ts;
            ts.set_name(edge_it->name());
            ts.set_shared(edge_it->shared());
            ts.set_share_from(edge_it->share_from());
            (*edges_info)[edge_it->name()].CopyFrom(ts);
        }

        for (auto& edge_it : arcs_it_out) {
            (*edges_out)[edge_it->first()].add_val(edge_it->second());
            TensorProto ts;
            ts.set_name(edge_it->name());
            ts.set_shared(edge_it->shared());
            ts.set_share_from(edge_it->share_from());
            (*edges_info)[edge_it->name()].CopyFrom(ts);
        }
    };

    graph->Scanner->BFS(insert_edge);


    // save graph info
    auto summary = graph_proto.mutable_summary();
    summary->set_is_optimized(graph->statistics.template get_info<graph::IS_OPTIMIZED>());
    summary->set_temp_mem_used(graph->statistics.template get_info<graph::TEMP_MEM>());
    summary->set_original_temp_mem_used(graph->statistics.template get_info<graph::ORI_TEMP_MEM>());
    summary->set_system_mem_used(graph->statistics.template get_info<graph::SYSTEM_MEM>());
    summary->set_model_mem_used(graph->statistics.template get_info<graph::MODEL_MEM>());

    //  save graph proto to disk
    graph_proto.SerializeToOstream(&output);

    return Status::OK();
}
#endif

#ifdef USE_CUDA
template
Status load<NV, AK_FLOAT, Precision::FP32>(graph::Graph<NV, AK_FLOAT, Precision::FP32>* graph,
        std::istream* instream);
template
Status load<NV, AK_FLOAT, Precision::FP16>(graph::Graph<NV, AK_FLOAT, Precision::FP16>* graph,
        std::istream* instream);
template
Status load<NV, AK_FLOAT, Precision::INT8>(graph::Graph<NV, AK_FLOAT, Precision::INT8>* graph,
        std::istream* instream);

template
Status load<NV, AK_FLOAT, Precision::FP32>(graph::Graph<NV, AK_FLOAT, Precision::FP32>* graph,
        const char* model_path);
template
Status load<NV, AK_FLOAT, Precision::FP16>(graph::Graph<NV, AK_FLOAT, Precision::FP16>* graph,
        const char* model_path);
template
Status load<NV, AK_FLOAT, Precision::INT8>(graph::Graph<NV, AK_FLOAT, Precision::INT8>* graph,
        const char* model_path);

template
Status save<NV, AK_FLOAT, Precision::FP32>(graph::Graph<NV, AK_FLOAT, Precision::FP32>* graph,
        std::string& model_path);
template
Status save<NV, AK_FLOAT, Precision::FP16>(graph::Graph<NV, AK_FLOAT, Precision::FP16>* graph,
        std::string& model_path);
template
Status save<NV, AK_FLOAT, Precision::INT8>(graph::Graph<NV, AK_FLOAT, Precision::INT8>* graph,
        std::string& model_path);

template
Status load<NV, AK_FLOAT, Precision::FP32>(graph::Graph<NV, AK_FLOAT, Precision::FP32>* graph,
        std::string& model_path);
template
Status load<NV, AK_FLOAT, Precision::FP16>(graph::Graph<NV, AK_FLOAT, Precision::FP16>* graph,
        std::string& model_path);
template
Status load<NV, AK_FLOAT, Precision::INT8>(graph::Graph<NV, AK_FLOAT, Precision::INT8>* graph,
        std::string& model_path);

template
Status save<NV, AK_FLOAT, Precision::FP32>(graph::Graph<NV, AK_FLOAT, Precision::FP32>* graph,
        const char* model_path);
template
Status save<NV, AK_FLOAT, Precision::FP16>(graph::Graph<NV, AK_FLOAT, Precision::FP16>* graph,
        const char* model_path);
template
Status save<NV, AK_FLOAT, Precision::INT8>(graph::Graph<NV, AK_FLOAT, Precision::INT8>* graph,
        const char* model_path);

template
Status load<NV, AK_FLOAT, Precision::FP32>(graph::Graph<NV, AK_FLOAT, Precision::FP32>* graph,
        const char* buffer, size_t len);
template
Status load<NV, AK_FLOAT, Precision::FP16>(graph::Graph<NV, AK_FLOAT, Precision::FP16>* graph,
        const char* buffer, size_t len);
template
Status load<NV, AK_FLOAT, Precision::INT8>(graph::Graph<NV, AK_FLOAT, Precision::INT8>* graph,
        const char* buffer, size_t len);

template
Status generate_graph_with_graph_proto<NV, AK_FLOAT, Precision::FP32>(graph::Graph<NV, AK_FLOAT, Precision::FP32>* graph, GraphProto& graph_proto);

template
Status generate_graph_with_graph_proto<NV, AK_FLOAT, Precision::FP16>(graph::Graph<NV, AK_FLOAT, Precision::FP16>* graph, GraphProto& graph_proto);

template
Status generate_graph_with_graph_proto<NV, AK_FLOAT, Precision::INT8>(graph::Graph<NV, AK_FLOAT, Precision::INT8>* graph, GraphProto& graph_proto);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
#ifndef USE_NANOPB
template
Status load<X86, AK_FLOAT, Precision::FP32>(graph::Graph<X86, AK_FLOAT, Precision::FP32>* graph,
        std::istream* instream);
template
Status load<X86, AK_FLOAT, Precision::FP16>(graph::Graph<X86, AK_FLOAT, Precision::FP16>* graph,
        std::istream* instream);
template
Status load<X86, AK_FLOAT, Precision::INT8>(graph::Graph<X86, AK_FLOAT, Precision::INT8>* graph,
        std::istream* instream);
#endif

template
Status load<X86, AK_FLOAT, Precision::FP32>(graph::Graph<X86, AK_FLOAT, Precision::FP32>* graph,
        const char* model_path);
template
Status load<X86, AK_FLOAT, Precision::FP16>(graph::Graph<X86, AK_FLOAT, Precision::FP16>* graph,
        const char* model_path);
template
Status load<X86, AK_FLOAT, Precision::INT8>(graph::Graph<X86, AK_FLOAT, Precision::INT8>* graph,
        const char* model_path);

#ifndef USE_NANOPB
template
Status save<X86, AK_FLOAT, Precision::FP32>(graph::Graph<X86, AK_FLOAT, Precision::FP32>* graph,
        std::string& model_path);
template
Status save<X86, AK_FLOAT, Precision::FP16>(graph::Graph<X86, AK_FLOAT, Precision::FP16>* graph,
        std::string& model_path);
template
Status save<X86, AK_FLOAT, Precision::INT8>(graph::Graph<X86, AK_FLOAT, Precision::INT8>* graph,
        std::string& model_path);
#endif

template
Status load<X86, AK_FLOAT, Precision::FP32>(graph::Graph<X86, AK_FLOAT, Precision::FP32>* graph,
        std::string& model_path);
template
Status load<X86, AK_FLOAT, Precision::FP16>(graph::Graph<X86, AK_FLOAT, Precision::FP16>* graph,
        std::string& model_path);
template
Status load<X86, AK_FLOAT, Precision::INT8>(graph::Graph<X86, AK_FLOAT, Precision::INT8>* graph,
        std::string& model_path);

#ifndef USE_NANOPB
template
Status save<X86, AK_FLOAT, Precision::FP32>(graph::Graph<X86, AK_FLOAT, Precision::FP32>* graph,
        const char* model_path);
template
Status save<X86, AK_FLOAT, Precision::FP16>(graph::Graph<X86, AK_FLOAT, Precision::FP16>* graph,
        const char* model_path);
template
Status save<X86, AK_FLOAT, Precision::INT8>(graph::Graph<X86, AK_FLOAT, Precision::INT8>* graph,
        const char* model_path);
#endif

template
Status load<X86, AK_FLOAT, Precision::FP32>(graph::Graph<X86, AK_FLOAT, Precision::FP32>* graph,
        const char* buffer, size_t len);
template
Status load<X86, AK_FLOAT, Precision::FP16>(graph::Graph<X86, AK_FLOAT, Precision::FP16>* graph,
        const char* buffer, size_t len);
template
Status load<X86, AK_FLOAT, Precision::INT8>(graph::Graph<X86, AK_FLOAT, Precision::INT8>* graph,
        const char* buffer, size_t len);

template
Status generate_graph_with_graph_proto<X86, AK_FLOAT, Precision::FP32>(graph::Graph<X86, AK_FLOAT, Precision::FP32>* graph, GraphProto& graph_proto);
template
Status generate_graph_with_graph_proto<X86, AK_FLOAT, Precision::FP16>(graph::Graph<X86, AK_FLOAT, Precision::FP16>* graph, GraphProto& graph_proto);
template
Status generate_graph_with_graph_proto<X86, AK_FLOAT, Precision::INT8>(graph::Graph<X86, AK_FLOAT, Precision::INT8>* graph, GraphProto& graph_proto);
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template
Status load<ARM, AK_FLOAT, Precision::FP32>(graph::Graph<ARM, AK_FLOAT, Precision::FP32>* graph,
                                                    std::istream* instream);
template
Status load<ARM, AK_FLOAT, Precision::FP32>(graph::Graph<ARM, AK_FLOAT, Precision::FP32>* graph,
                                                    const char* model_path);
template
Status save<ARM, AK_FLOAT, Precision::FP32>(graph::Graph<ARM, AK_FLOAT, Precision::FP32>* graph,
                                                    std::string& model_path);
template
Status load<ARM, AK_FLOAT, Precision::FP32>(graph::Graph<ARM, AK_FLOAT, Precision::FP32>* graph,
                                                    std::string& model_path);
template
Status save<ARM, AK_FLOAT, Precision::FP32>(graph::Graph<ARM, AK_FLOAT, Precision::FP32>* graph,
                                                    const char* model_path);

template
Status load<ARM, AK_FLOAT, Precision::FP32>(graph::Graph<ARM, AK_FLOAT, Precision::FP32>* graph,
                                                    const char* buffer, size_t len);

template
Status generate_graph_with_graph_proto<ARM, AK_FLOAT, Precision::FP32>(graph::Graph<ARM, AK_FLOAT, Precision::FP32>* graph, GraphProto& graph_proto);
#endif

#ifdef ANAKIN_TYPE_FP16
template
Status load<ARM, AK_FLOAT, Precision::FP16>(graph::Graph<ARM, AK_FLOAT, Precision::FP16>* graph,
                                                    std::istream* instream);

template
Status load<ARM, AK_FLOAT, Precision::FP16>(graph::Graph<ARM, AK_FLOAT, Precision::FP16>* graph,
                                                    const char* model_path);
template
Status save<ARM, AK_FLOAT, Precision::FP16>(graph::Graph<ARM, AK_FLOAT, Precision::FP16>* graph,
                                                    std::string& model_path);
template
Status load<ARM, AK_FLOAT, Precision::FP16>(graph::Graph<ARM, AK_FLOAT, Precision::FP16>* graph,
                                                    std::string& model_path);
template
Status save<ARM, AK_FLOAT, Precision::FP16>(graph::Graph<ARM, AK_FLOAT, Precision::FP16>* graph,
                                                    const char* model_path);

template
Status load<ARM, AK_FLOAT, Precision::FP16>(graph::Graph<ARM, AK_FLOAT, Precision::FP16>* graph,
                                                    const char* buffer, size_t len);

template
Status generate_graph_with_graph_proto<ARM, AK_FLOAT, Precision::FP16>(graph::Graph<ARM, AK_FLOAT, Precision::FP16>* graph, GraphProto& graph_proto);
#endif

#ifdef ANAKIN_TYPE_INT8
template
Status load<ARM, AK_FLOAT, Precision::INT8>(graph::Graph<ARM, AK_FLOAT, Precision::INT8>* graph,
                                                    std::istream* instream);
template
Status load<ARM, AK_FLOAT, Precision::INT8>(graph::Graph<ARM, AK_FLOAT, Precision::INT8>* graph,
                                                    const char* model_path);
template
Status save<ARM, AK_FLOAT, Precision::INT8>(graph::Graph<ARM, AK_FLOAT, Precision::INT8>* graph,
                                                    std::string& model_path);
template
Status load<ARM, AK_FLOAT, Precision::INT8>(graph::Graph<ARM, AK_FLOAT, Precision::INT8>* graph,
                                                    std::string& model_path);
template
Status save<ARM, AK_FLOAT, Precision::INT8>(graph::Graph<ARM, AK_FLOAT, Precision::INT8>* graph,
                                                    const char* model_path);

template
Status load<ARM, AK_FLOAT, Precision::INT8>(graph::Graph<ARM, AK_FLOAT, Precision::INT8>* graph,
                                                    const char* buffer, size_t len);

template
Status generate_graph_with_graph_proto<ARM, AK_FLOAT, Precision::INT8>(graph::Graph<ARM, AK_FLOAT, Precision::INT8>* graph, GraphProto& graph_proto);
#endif

#endif //USE_ARM_PLACE

} /* parser */

} /* anakin */
