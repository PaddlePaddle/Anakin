#include <pb_decode.h>

#include <algorithm>
#include <memory>

#include "graph.pb.hpp"

#include <pb_cpp_decode.h>

namespace nanopb_cpp {

void Version::fill(Nanopb *pb) {

    // major: optional int32
    
    // minor: optional int32
    
    // patch: optional int32
    
    // version: optional int64
    
}

void Version::retrieve(const Nanopb *pb) {

    // major: optional int32
    _major = static_cast<decltype(_major)>(pb->major);
    
    // minor: optional int32
    _minor = static_cast<decltype(_minor)>(pb->minor);
    
    // patch: optional int32
    _patch = static_cast<decltype(_patch)>(pb->patch);
    
    // version: optional int64
    _version = static_cast<decltype(_version)>(pb->version);
    
}

IMPLEMENT_PARSING_WRAPPERS(Version);

void Info::fill(Nanopb *pb) {

    // temp_mem_used: optional int32
    
    // original_temp_mem_used: optional int32
    
    // system_mem_used: optional int32
    
    // model_mem_used: optional int32
    
    // is_optimized: optional bool
    
}

void Info::retrieve(const Nanopb *pb) {

    // temp_mem_used: optional int32
    _temp_mem_used = static_cast<decltype(_temp_mem_used)>(pb->temp_mem_used);
    
    // original_temp_mem_used: optional int32
    _original_temp_mem_used = static_cast<decltype(_original_temp_mem_used)>(pb->original_temp_mem_used);
    
    // system_mem_used: optional int32
    _system_mem_used = static_cast<decltype(_system_mem_used)>(pb->system_mem_used);
    
    // model_mem_used: optional int32
    _model_mem_used = static_cast<decltype(_model_mem_used)>(pb->model_mem_used);
    
    // is_optimized: optional bool
    _is_optimized = static_cast<decltype(_is_optimized)>(pb->is_optimized);
    
}

IMPLEMENT_PARSING_WRAPPERS(Info);

void TargetProto::fill(Nanopb *pb) {

    // node: optional string
    pb->node.funcs.decode = decode_string;
    pb->node.arg = &_node;
    
    // scale: repeated float
    pb->scale.funcs.decode = decode_repeated<float, decode_fixed32<float>>;
    pb->scale.arg = &_scale;
    
    // layout: optional LayoutProto
    
}

void TargetProto::retrieve(const Nanopb *pb) {

    // node: optional string
    
    // scale: repeated float
    
    // layout: optional LayoutProto
    _layout = static_cast<decltype(_layout)>(pb->layout);
    
}

IMPLEMENT_PARSING_WRAPPERS(TargetProto);

void List::fill(Nanopb *pb) {

    // val: repeated string
    pb->val.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->val.arg = &_val;
    
    // target: repeated TargetProto
    pb->target.funcs.decode = decode_repeated<nanopb_cpp::TargetProto, decode_message<nanopb_cpp::TargetProto>>;
    pb->target.arg = &_target;
    
}

void List::retrieve(const Nanopb *pb) {

    // val: repeated string
    
    // target: repeated TargetProto
    
}

IMPLEMENT_PARSING_WRAPPERS(List);

void GraphProto::EdgesInEntry::fill(Nanopb *pb) {

    // key: optional string
    pb->key.funcs.decode = decode_string;
    pb->key.arg = &_key;
    
    // value: optional List
    _value.fill(&pb->value);
    
}

void GraphProto::EdgesInEntry::retrieve(const Nanopb *pb) {

    // key: optional string
    
    // value: optional List
    _value.retrieve(&pb->value);
    
}


void GraphProto::EdgesOutEntry::fill(Nanopb *pb) {

    // key: optional string
    pb->key.funcs.decode = decode_string;
    pb->key.arg = &_key;
    
    // value: optional List
    _value.fill(&pb->value);
    
}

void GraphProto::EdgesOutEntry::retrieve(const Nanopb *pb) {

    // key: optional string
    
    // value: optional List
    _value.retrieve(&pb->value);
    
}


void GraphProto::EdgesInfoEntry::fill(Nanopb *pb) {

    // key: optional string
    pb->key.funcs.decode = decode_string;
    pb->key.arg = &_key;
    
    // value: optional TensorProto
    _value.fill(&pb->value);
    
}

void GraphProto::EdgesInfoEntry::retrieve(const Nanopb *pb) {

    // key: optional string
    
    // value: optional TensorProto
    _value.retrieve(&pb->value);
    
}


void GraphProto::fill(Nanopb *pb) {

    // name: optional string
    pb->name.funcs.decode = decode_string;
    pb->name.arg = &_name;
    
    // nodes: repeated NodeProto
    pb->nodes.funcs.decode = decode_repeated<nanopb_cpp::NodeProto, decode_message<nanopb_cpp::NodeProto>>;
    pb->nodes.arg = &_nodes;
    
    // edges_in: repeated GraphProto.EdgesInEntry
    pb->edges_in.funcs.decode = decode_map<nanopb_cpp::GraphProto::EdgesInEntry>;
    pb->edges_in.arg = &_edges_in;
    
    // edges_out: repeated GraphProto.EdgesOutEntry
    pb->edges_out.funcs.decode = decode_map<nanopb_cpp::GraphProto::EdgesOutEntry>;
    pb->edges_out.arg = &_edges_out;
    
    // edges_info: repeated GraphProto.EdgesInfoEntry
    pb->edges_info.funcs.decode = decode_map<nanopb_cpp::GraphProto::EdgesInfoEntry>;
    pb->edges_info.arg = &_edges_info;
    
    // ins: repeated string
    pb->ins.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->ins.arg = &_ins;
    
    // outs: repeated string
    pb->outs.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->outs.arg = &_outs;
    
    // version: optional Version
    _version.fill(&pb->version);
    
    // summary: optional Info
    _summary.fill(&pb->summary);
    
}

void GraphProto::retrieve(const Nanopb *pb) {

    // name: optional string
    
    // nodes: repeated NodeProto
    
    // edges_in: repeated GraphProto.EdgesInEntry
    
    // edges_out: repeated GraphProto.EdgesOutEntry
    
    // edges_info: repeated GraphProto.EdgesInfoEntry
    
    // ins: repeated string
    
    // outs: repeated string
    
    // version: optional Version
    _version.retrieve(&pb->version);
    
    // summary: optional Info
    _summary.retrieve(&pb->summary);
    
}

IMPLEMENT_PARSING_WRAPPERS(GraphProto);

} // namespace nanopb_cpp
