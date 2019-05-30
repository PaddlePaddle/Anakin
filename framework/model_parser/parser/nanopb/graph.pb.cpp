#include "graph.pb.hpp"
#include <pb_codec.hpp>

namespace nanopb_cpp {

void Version::CopyFrom(const Version &other) {

    // major: optional int32
    field_copy(_major, other._major);

    // minor: optional int32
    field_copy(_minor, other._minor);

    // patch: optional int32
    field_copy(_patch, other._patch);

    // version: optional int64
    field_copy(_version, other._version);

}

void Version::pre_decode(Nanopb *pb) {

    // major: optional int32
    
    // minor: optional int32
    
    // patch: optional int32
    
    // version: optional int64
    
}

void Version::post_decode(const Nanopb *pb) {

    // major: optional int32
    _major = static_cast<decltype(_major)>(pb->major);
    
    // minor: optional int32
    _minor = static_cast<decltype(_minor)>(pb->minor);
    
    // patch: optional int32
    _patch = static_cast<decltype(_patch)>(pb->patch);
    
    // version: optional int64
    _version = static_cast<decltype(_version)>(pb->version);
    
}

void Version::pre_encode(Nanopb *pb) const {

    // major: optional int32
    pb->major = static_cast<decltype(pb->major)>(_major);
    
    // minor: optional int32
    pb->minor = static_cast<decltype(pb->minor)>(_minor);
    
    // patch: optional int32
    pb->patch = static_cast<decltype(pb->patch)>(_patch);
    
    // version: optional int64
    pb->version = static_cast<decltype(pb->version)>(_version);
    
}

IMPLEMENT_CODEC_MEMBERS(Version);
void Info::CopyFrom(const Info &other) {

    // temp_mem_used: optional int32
    field_copy(_temp_mem_used, other._temp_mem_used);

    // original_temp_mem_used: optional int32
    field_copy(_original_temp_mem_used, other._original_temp_mem_used);

    // system_mem_used: optional int32
    field_copy(_system_mem_used, other._system_mem_used);

    // model_mem_used: optional int32
    field_copy(_model_mem_used, other._model_mem_used);

    // is_optimized: optional bool
    field_copy(_is_optimized, other._is_optimized);

}

void Info::pre_decode(Nanopb *pb) {

    // temp_mem_used: optional int32
    
    // original_temp_mem_used: optional int32
    
    // system_mem_used: optional int32
    
    // model_mem_used: optional int32
    
    // is_optimized: optional bool
    
}

void Info::post_decode(const Nanopb *pb) {

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

void Info::pre_encode(Nanopb *pb) const {

    // temp_mem_used: optional int32
    pb->temp_mem_used = static_cast<decltype(pb->temp_mem_used)>(_temp_mem_used);
    
    // original_temp_mem_used: optional int32
    pb->original_temp_mem_used = static_cast<decltype(pb->original_temp_mem_used)>(_original_temp_mem_used);
    
    // system_mem_used: optional int32
    pb->system_mem_used = static_cast<decltype(pb->system_mem_used)>(_system_mem_used);
    
    // model_mem_used: optional int32
    pb->model_mem_used = static_cast<decltype(pb->model_mem_used)>(_model_mem_used);
    
    // is_optimized: optional bool
    pb->is_optimized = static_cast<decltype(pb->is_optimized)>(_is_optimized);
    
}

IMPLEMENT_CODEC_MEMBERS(Info);
void TargetProto::CopyFrom(const TargetProto &other) {

    // node: optional string
    field_copy(_node, other._node);

    // scale: repeated float
    field_copy(_scale, other._scale);

    // layout: optional LayoutProto
    field_copy(_layout, other._layout);

}

void TargetProto::pre_decode(Nanopb *pb) {

    // node: optional string
    pb->node.funcs.decode = codec_obj<std::string>::decode;
    pb->node.arg = &_node;
    
    // scale: repeated float
    pb->scale.funcs.decode = codec_repeat<float, codec_fixed32>::decode;
    pb->scale.arg = &_scale;
    
    // layout: optional LayoutProto
    
}

void TargetProto::post_decode(const Nanopb *pb) {

    // node: optional string
    
    // scale: repeated float
    
    // layout: optional LayoutProto
    _layout = static_cast<decltype(_layout)>(pb->layout);
    
}

void TargetProto::pre_encode(Nanopb *pb) const {

    // node: optional string
    pb->node.funcs.encode = codec_obj<std::string>::encode;
    pb->node.arg = const_cast<void *>(static_cast<const void *>(&_node));
    
    // scale: repeated float
    pb->scale.funcs.encode = codec_repeat<float, codec_fixed32>::encode;
    pb->scale.arg = const_cast<void *>(static_cast<const void *>(&_scale));
    
    // layout: optional LayoutProto
    pb->layout = static_cast<decltype(pb->layout)>(_layout);
    
}

IMPLEMENT_CODEC_MEMBERS(TargetProto);
void List::CopyFrom(const List &other) {

    // val: repeated string
    field_copy(_val, other._val);

    // target: repeated TargetProto
    field_copy(_target, other._target);

}

void List::pre_decode(Nanopb *pb) {

    // val: repeated string
    pb->val.funcs.decode = codec_repeat<std::string, codec_obj>::decode;
    pb->val.arg = &_val;
    
    // target: repeated TargetProto
    pb->target.funcs.decode = codec_repeat<nanopb_cpp::TargetProto, codec_obj>::decode;
    pb->target.arg = &_target;
    
}

void List::post_decode(const Nanopb *pb) {

    // val: repeated string
    
    // target: repeated TargetProto
    
}

void List::pre_encode(Nanopb *pb) const {

    // val: repeated string
    pb->val.funcs.encode = codec_repeat<std::string, codec_obj>::encode;
    pb->val.arg = const_cast<void *>(static_cast<const void *>(&_val));
    
    // target: repeated TargetProto
    pb->target.funcs.encode = codec_repeat<nanopb_cpp::TargetProto, codec_obj>::encode;
    pb->target.arg = const_cast<void *>(static_cast<const void *>(&_target));
    
}

IMPLEMENT_CODEC_MEMBERS(List);
void GraphProto::EdgesInEntry::CopyFrom(const GraphProto::EdgesInEntry &other) {

    // key: optional string
    field_copy(_key, other._key);

    // value: optional List
    field_copy(_value, other._value);

}

void GraphProto::EdgesInEntry::pre_decode(Nanopb *pb) {

    // key: optional string
    pb->key.funcs.decode = codec_obj<std::string>::decode;
    pb->key.arg = &_key;
    
    // value: optional List
    _value->pre_decode(&pb->value);
    
}

void GraphProto::EdgesInEntry::post_decode(const Nanopb *pb) {

    // key: optional string
    
    // value: optional List
    _value->post_decode(&pb->value);
    
}

void GraphProto::EdgesInEntry::pre_encode(Nanopb *pb) const {

    // key: optional string
    pb->key.funcs.encode = codec_obj<std::string>::encode;
    pb->key.arg = const_cast<void *>(static_cast<const void *>(&_key));
    
    // value: optional List
    _value->pre_encode(&pb->value);
    
}

IMPLEMENT_CODEC_MEMBERS(GraphProto::EdgesInEntry);
void GraphProto::EdgesOutEntry::CopyFrom(const GraphProto::EdgesOutEntry &other) {

    // key: optional string
    field_copy(_key, other._key);

    // value: optional List
    field_copy(_value, other._value);

}

void GraphProto::EdgesOutEntry::pre_decode(Nanopb *pb) {

    // key: optional string
    pb->key.funcs.decode = codec_obj<std::string>::decode;
    pb->key.arg = &_key;
    
    // value: optional List
    _value->pre_decode(&pb->value);
    
}

void GraphProto::EdgesOutEntry::post_decode(const Nanopb *pb) {

    // key: optional string
    
    // value: optional List
    _value->post_decode(&pb->value);
    
}

void GraphProto::EdgesOutEntry::pre_encode(Nanopb *pb) const {

    // key: optional string
    pb->key.funcs.encode = codec_obj<std::string>::encode;
    pb->key.arg = const_cast<void *>(static_cast<const void *>(&_key));
    
    // value: optional List
    _value->pre_encode(&pb->value);
    
}

IMPLEMENT_CODEC_MEMBERS(GraphProto::EdgesOutEntry);
void GraphProto::EdgesInfoEntry::CopyFrom(const GraphProto::EdgesInfoEntry &other) {

    // key: optional string
    field_copy(_key, other._key);

    // value: optional TensorProto
    field_copy(_value, other._value);

}

void GraphProto::EdgesInfoEntry::pre_decode(Nanopb *pb) {

    // key: optional string
    pb->key.funcs.decode = codec_obj<std::string>::decode;
    pb->key.arg = &_key;
    
    // value: optional TensorProto
    _value->pre_decode(&pb->value);
    
}

void GraphProto::EdgesInfoEntry::post_decode(const Nanopb *pb) {

    // key: optional string
    
    // value: optional TensorProto
    _value->post_decode(&pb->value);
    
}

void GraphProto::EdgesInfoEntry::pre_encode(Nanopb *pb) const {

    // key: optional string
    pb->key.funcs.encode = codec_obj<std::string>::encode;
    pb->key.arg = const_cast<void *>(static_cast<const void *>(&_key));
    
    // value: optional TensorProto
    _value->pre_encode(&pb->value);
    
}

IMPLEMENT_CODEC_MEMBERS(GraphProto::EdgesInfoEntry);
void GraphProto::CopyFrom(const GraphProto &other) {

    // name: optional string
    field_copy(_name, other._name);

    // nodes: repeated NodeProto
    field_copy(_nodes, other._nodes);

    // edges_in: repeated GraphProto.EdgesInEntry
    field_copy(_edges_in, other._edges_in);

    // edges_out: repeated GraphProto.EdgesOutEntry
    field_copy(_edges_out, other._edges_out);

    // edges_info: repeated GraphProto.EdgesInfoEntry
    field_copy(_edges_info, other._edges_info);

    // ins: repeated string
    field_copy(_ins, other._ins);

    // outs: repeated string
    field_copy(_outs, other._outs);

    // version: optional Version
    field_copy(_version, other._version);

    // summary: optional Info
    field_copy(_summary, other._summary);

}

void GraphProto::pre_decode(Nanopb *pb) {

    // name: optional string
    pb->name.funcs.decode = codec_obj<std::string>::decode;
    pb->name.arg = &_name;
    
    // nodes: repeated NodeProto
    pb->nodes.funcs.decode = codec_repeat<nanopb_cpp::NodeProto, codec_obj>::decode;
    pb->nodes.arg = &_nodes;
    
    // edges_in: repeated GraphProto.EdgesInEntry
    pb->edges_in.funcs.decode = codec_map<nanopb_cpp::GraphProto::EdgesInEntry>::decode;
    pb->edges_in.arg = &_edges_in;
    
    // edges_out: repeated GraphProto.EdgesOutEntry
    pb->edges_out.funcs.decode = codec_map<nanopb_cpp::GraphProto::EdgesOutEntry>::decode;
    pb->edges_out.arg = &_edges_out;
    
    // edges_info: repeated GraphProto.EdgesInfoEntry
    pb->edges_info.funcs.decode = codec_map<nanopb_cpp::GraphProto::EdgesInfoEntry>::decode;
    pb->edges_info.arg = &_edges_info;
    
    // ins: repeated string
    pb->ins.funcs.decode = codec_repeat<std::string, codec_obj>::decode;
    pb->ins.arg = &_ins;
    
    // outs: repeated string
    pb->outs.funcs.decode = codec_repeat<std::string, codec_obj>::decode;
    pb->outs.arg = &_outs;
    
    // version: optional Version
    _version->pre_decode(&pb->version);
    
    // summary: optional Info
    _summary->pre_decode(&pb->summary);
    
}

void GraphProto::post_decode(const Nanopb *pb) {

    // name: optional string
    
    // nodes: repeated NodeProto
    
    // edges_in: repeated GraphProto.EdgesInEntry
    
    // edges_out: repeated GraphProto.EdgesOutEntry
    
    // edges_info: repeated GraphProto.EdgesInfoEntry
    
    // ins: repeated string
    
    // outs: repeated string
    
    // version: optional Version
    _version->post_decode(&pb->version);
    
    // summary: optional Info
    _summary->post_decode(&pb->summary);
    
}

void GraphProto::pre_encode(Nanopb *pb) const {

    // name: optional string
    pb->name.funcs.encode = codec_obj<std::string>::encode;
    pb->name.arg = const_cast<void *>(static_cast<const void *>(&_name));
    
    // nodes: repeated NodeProto
    pb->nodes.funcs.encode = codec_repeat<nanopb_cpp::NodeProto, codec_obj>::encode;
    pb->nodes.arg = const_cast<void *>(static_cast<const void *>(&_nodes));
    
    // edges_in: repeated GraphProto.EdgesInEntry
    pb->edges_in.funcs.encode = codec_map<nanopb_cpp::GraphProto::EdgesInEntry>::encode;
    pb->edges_in.arg = const_cast<void *>(static_cast<const void *>(&_edges_in));
    
    // edges_out: repeated GraphProto.EdgesOutEntry
    pb->edges_out.funcs.encode = codec_map<nanopb_cpp::GraphProto::EdgesOutEntry>::encode;
    pb->edges_out.arg = const_cast<void *>(static_cast<const void *>(&_edges_out));
    
    // edges_info: repeated GraphProto.EdgesInfoEntry
    pb->edges_info.funcs.encode = codec_map<nanopb_cpp::GraphProto::EdgesInfoEntry>::encode;
    pb->edges_info.arg = const_cast<void *>(static_cast<const void *>(&_edges_info));
    
    // ins: repeated string
    pb->ins.funcs.encode = codec_repeat<std::string, codec_obj>::encode;
    pb->ins.arg = const_cast<void *>(static_cast<const void *>(&_ins));
    
    // outs: repeated string
    pb->outs.funcs.encode = codec_repeat<std::string, codec_obj>::encode;
    pb->outs.arg = const_cast<void *>(static_cast<const void *>(&_outs));
    
    // version: optional Version
    _version->pre_encode(&pb->version);
    
    // summary: optional Info
    _summary->pre_encode(&pb->summary);
    
}

IMPLEMENT_CODEC_MEMBERS(GraphProto);
} // namespace nanopb_cpp
