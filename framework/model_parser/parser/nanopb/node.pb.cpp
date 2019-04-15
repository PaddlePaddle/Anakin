#include <pb_decode.h>

#include <algorithm>
#include <memory>

#include "node.pb.hpp"

#include <pb_cpp_decode.h>

namespace nanopb_cpp {

void valueType::fill(Nanopb *pb) {

    // s: optional bytes
    pb->s.funcs.decode = decode_string;
    pb->s.arg = &_s;
    
    // i: optional int32
    
    // f: optional float
    
    // b: optional bool
    
    // cache_list: optional CacheDate
    _cache_list.fill(&pb->cache_list);
    
    // tensor: optional TensorProto
    _tensor.fill(&pb->tensor);
    
    // type: optional DateTypeProto
    
}

void valueType::retrieve(const Nanopb *pb) {

    // s: optional bytes
    
    // i: optional int32
    _i = static_cast<decltype(_i)>(pb->i);
    
    // f: optional float
    _f = static_cast<decltype(_f)>(pb->f);
    
    // b: optional bool
    _b = static_cast<decltype(_b)>(pb->b);
    
    // cache_list: optional CacheDate
    _cache_list.retrieve(&pb->cache_list);
    
    // tensor: optional TensorProto
    _tensor.retrieve(&pb->tensor);
    
    // type: optional DateTypeProto
    _type = static_cast<decltype(_type)>(pb->type);
    
}

IMPLEMENT_PARSING_WRAPPERS(valueType);

void NodeProto::AttrEntry::fill(Nanopb *pb) {

    // key: optional string
    pb->key.funcs.decode = decode_string;
    pb->key.arg = &_key;
    
    // value: optional valueType
    _value.fill(&pb->value);
    
}

void NodeProto::AttrEntry::retrieve(const Nanopb *pb) {

    // key: optional string
    
    // value: optional valueType
    _value.retrieve(&pb->value);
    
}


void NodeProto::fill(Nanopb *pb) {

    // name: optional string
    pb->name.funcs.decode = decode_string;
    pb->name.arg = &_name;
    
    // ins: repeated string
    pb->ins.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->ins.arg = &_ins;
    
    // outs: repeated string
    pb->outs.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->outs.arg = &_outs;
    
    // attr: repeated NodeProto.AttrEntry
    pb->attr.funcs.decode = decode_map<nanopb_cpp::NodeProto::AttrEntry>;
    pb->attr.arg = &_attr;
    
    // lane: optional int32
    
    // need_wait: optional bool
    
    // Op: optional OpProto
    _op.fill(&pb->Op);
    
    // bit_type: optional DateTypeProto
    
}

void NodeProto::retrieve(const Nanopb *pb) {

    // name: optional string
    
    // ins: repeated string
    
    // outs: repeated string
    
    // attr: repeated NodeProto.AttrEntry
    
    // lane: optional int32
    _lane = static_cast<decltype(_lane)>(pb->lane);
    
    // need_wait: optional bool
    _need_wait = static_cast<decltype(_need_wait)>(pb->need_wait);
    
    // Op: optional OpProto
    _op.retrieve(&pb->Op);
    
    // bit_type: optional DateTypeProto
    _bit_type = static_cast<decltype(_bit_type)>(pb->bit_type);
    
}

IMPLEMENT_PARSING_WRAPPERS(NodeProto);

} // namespace nanopb_cpp
