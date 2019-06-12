#include "operator.pb.hpp"
#include <pb_codec.hpp>

namespace nanopb_cpp {

void OpProto::CopyFrom(const OpProto &other) {

    // name: optional string
    field_copy(_name, other._name);

    // is_commutative: optional bool
    field_copy(_is_commutative, other._is_commutative);

    // in_num: optional int32
    field_copy(_in_num, other._in_num);

    // out_num: optional int32
    field_copy(_out_num, other._out_num);

    // description: optional string
    field_copy(_description, other._description);

}

void OpProto::pre_decode(Nanopb *pb) {

    // name: optional string
    pb->name.funcs.decode = codec_obj<std::string>::decode;
    pb->name.arg = &_name;
    
    // is_commutative: optional bool
    
    // in_num: optional int32
    
    // out_num: optional int32
    
    // description: optional string
    pb->description.funcs.decode = codec_obj<std::string>::decode;
    pb->description.arg = &_description;
    
}

void OpProto::post_decode(const Nanopb *pb) {

    // name: optional string
    
    // is_commutative: optional bool
    _is_commutative = static_cast<decltype(_is_commutative)>(pb->is_commutative);
    
    // in_num: optional int32
    _in_num = static_cast<decltype(_in_num)>(pb->in_num);
    
    // out_num: optional int32
    _out_num = static_cast<decltype(_out_num)>(pb->out_num);
    
    // description: optional string
    
}

void OpProto::pre_encode(Nanopb *pb) const {

    // name: optional string
    pb->name.funcs.encode = codec_obj<std::string>::encode;
    pb->name.arg = const_cast<void *>(static_cast<const void *>(&_name));
    
    // is_commutative: optional bool
    pb->is_commutative = static_cast<decltype(pb->is_commutative)>(_is_commutative);
    
    // in_num: optional int32
    pb->in_num = static_cast<decltype(pb->in_num)>(_in_num);
    
    // out_num: optional int32
    pb->out_num = static_cast<decltype(pb->out_num)>(_out_num);
    
    // description: optional string
    pb->description.funcs.encode = codec_obj<std::string>::encode;
    pb->description.arg = const_cast<void *>(static_cast<const void *>(&_description));
    
}

IMPLEMENT_CODEC_MEMBERS(OpProto);
} // namespace nanopb_cpp
