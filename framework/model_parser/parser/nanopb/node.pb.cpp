#include "node.pb.hpp"
#include <pb_codec.hpp>

namespace nanopb_cpp {

void valueType::CopyFrom(const valueType &other) {

    // s: optional bytes
    field_copy(_s, other._s);

    // i: optional int32
    field_copy(_i, other._i);

    // f: optional float
    field_copy(_f, other._f);

    // b: optional bool
    field_copy(_b, other._b);

    // cache_list: optional CacheDate
    field_copy(_cache_list, other._cache_list);

    // tensor: optional TensorProto
    field_copy(_tensor, other._tensor);

    // type: optional DateTypeProto
    field_copy(_type, other._type);

}

void valueType::pre_decode(Nanopb *pb) {

    // s: optional bytes
    pb->s.funcs.decode = codec_obj<std::string>::decode;
    pb->s.arg = &_s;
    
    // i: optional int32
    
    // f: optional float
    
    // b: optional bool
    
    // cache_list: optional CacheDate
    _cache_list->pre_decode(&pb->cache_list);
    
    // tensor: optional TensorProto
    _tensor->pre_decode(&pb->tensor);
    
    // type: optional DateTypeProto
    
}

void valueType::post_decode(const Nanopb *pb) {

    // s: optional bytes
    
    // i: optional int32
    _i = static_cast<decltype(_i)>(pb->i);
    
    // f: optional float
    _f = static_cast<decltype(_f)>(pb->f);
    
    // b: optional bool
    _b = static_cast<decltype(_b)>(pb->b);
    
    // cache_list: optional CacheDate
    _cache_list->post_decode(&pb->cache_list);
    
    // tensor: optional TensorProto
    _tensor->post_decode(&pb->tensor);
    
    // type: optional DateTypeProto
    _type = static_cast<decltype(_type)>(pb->type);
    
}

void valueType::pre_encode(Nanopb *pb) const {

    // s: optional bytes
    pb->s.funcs.encode = codec_obj<std::string>::encode;
    pb->s.arg = const_cast<void *>(static_cast<const void *>(&_s));
    
    // i: optional int32
    pb->i = static_cast<decltype(pb->i)>(_i);
    
    // f: optional float
    pb->f = static_cast<decltype(pb->f)>(_f);
    
    // b: optional bool
    pb->b = static_cast<decltype(pb->b)>(_b);
    
    // cache_list: optional CacheDate
    _cache_list->pre_encode(&pb->cache_list);
    
    // tensor: optional TensorProto
    _tensor->pre_encode(&pb->tensor);
    
    // type: optional DateTypeProto
    pb->type = static_cast<decltype(pb->type)>(_type);
    
}

IMPLEMENT_CODEC_MEMBERS(valueType);
void NodeProto::AttrEntry::CopyFrom(const NodeProto::AttrEntry &other) {

    // key: optional string
    field_copy(_key, other._key);

    // value: optional valueType
    field_copy(_value, other._value);

}

void NodeProto::AttrEntry::pre_decode(Nanopb *pb) {

    // key: optional string
    pb->key.funcs.decode = codec_obj<std::string>::decode;
    pb->key.arg = &_key;
    
    // value: optional valueType
    _value->pre_decode(&pb->value);
    
}

void NodeProto::AttrEntry::post_decode(const Nanopb *pb) {

    // key: optional string
    
    // value: optional valueType
    _value->post_decode(&pb->value);
    
}

void NodeProto::AttrEntry::pre_encode(Nanopb *pb) const {

    // key: optional string
    pb->key.funcs.encode = codec_obj<std::string>::encode;
    pb->key.arg = const_cast<void *>(static_cast<const void *>(&_key));
    
    // value: optional valueType
    _value->pre_encode(&pb->value);
    
}

IMPLEMENT_CODEC_MEMBERS(NodeProto::AttrEntry);
void NodeProto::CopyFrom(const NodeProto &other) {

    // name: optional string
    field_copy(_name, other._name);

    // ins: repeated string
    field_copy(_ins, other._ins);

    // outs: repeated string
    field_copy(_outs, other._outs);

    // attr: repeated NodeProto.AttrEntry
    field_copy(_attr, other._attr);

    // lane: optional int32
    field_copy(_lane, other._lane);

    // need_wait: optional bool
    field_copy(_need_wait, other._need_wait);

    // Op: optional OpProto
    field_copy(_op, other._op);

    // bit_type: optional DateTypeProto
    field_copy(_bit_type, other._bit_type);

}

void NodeProto::pre_decode(Nanopb *pb) {

    // name: optional string
    pb->name.funcs.decode = codec_obj<std::string>::decode;
    pb->name.arg = &_name;
    
    // ins: repeated string
    pb->ins.funcs.decode = codec_repeat<std::string, codec_obj>::decode;
    pb->ins.arg = &_ins;
    
    // outs: repeated string
    pb->outs.funcs.decode = codec_repeat<std::string, codec_obj>::decode;
    pb->outs.arg = &_outs;
    
    // attr: repeated NodeProto.AttrEntry
    pb->attr.funcs.decode = codec_map<nanopb_cpp::NodeProto::AttrEntry>::decode;
    pb->attr.arg = &_attr;
    
    // lane: optional int32
    
    // need_wait: optional bool
    
    // Op: optional OpProto
    _op->pre_decode(&pb->Op);
    
    // bit_type: optional DateTypeProto
    
}

void NodeProto::post_decode(const Nanopb *pb) {

    // name: optional string
    
    // ins: repeated string
    
    // outs: repeated string
    
    // attr: repeated NodeProto.AttrEntry
    
    // lane: optional int32
    _lane = static_cast<decltype(_lane)>(pb->lane);
    
    // need_wait: optional bool
    _need_wait = static_cast<decltype(_need_wait)>(pb->need_wait);
    
    // Op: optional OpProto
    _op->post_decode(&pb->Op);
    
    // bit_type: optional DateTypeProto
    _bit_type = static_cast<decltype(_bit_type)>(pb->bit_type);
    
}

void NodeProto::pre_encode(Nanopb *pb) const {

    // name: optional string
    pb->name.funcs.encode = codec_obj<std::string>::encode;
    pb->name.arg = const_cast<void *>(static_cast<const void *>(&_name));
    
    // ins: repeated string
    pb->ins.funcs.encode = codec_repeat<std::string, codec_obj>::encode;
    pb->ins.arg = const_cast<void *>(static_cast<const void *>(&_ins));
    
    // outs: repeated string
    pb->outs.funcs.encode = codec_repeat<std::string, codec_obj>::encode;
    pb->outs.arg = const_cast<void *>(static_cast<const void *>(&_outs));
    
    // attr: repeated NodeProto.AttrEntry
    pb->attr.funcs.encode = codec_map<nanopb_cpp::NodeProto::AttrEntry>::encode;
    pb->attr.arg = const_cast<void *>(static_cast<const void *>(&_attr));
    
    // lane: optional int32
    pb->lane = static_cast<decltype(pb->lane)>(_lane);
    
    // need_wait: optional bool
    pb->need_wait = static_cast<decltype(pb->need_wait)>(_need_wait);
    
    // Op: optional OpProto
    _op->pre_encode(&pb->Op);
    
    // bit_type: optional DateTypeProto
    pb->bit_type = static_cast<decltype(pb->bit_type)>(_bit_type);
    
}

IMPLEMENT_CODEC_MEMBERS(NodeProto);
} // namespace nanopb_cpp
