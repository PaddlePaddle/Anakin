#include <pb_decode.h>

#include <algorithm>
#include <memory>

#include "operator.pb.hpp"

#include <pb_cpp_decode.h>

namespace nanopb_cpp {

void OpProto::fill(Nanopb *pb) {

    // name: optional string
    pb->name.funcs.decode = decode_string;
    pb->name.arg = &_name;
    
    // is_commutative: optional bool
    
    // in_num: optional int32
    
    // out_num: optional int32
    
    // description: optional string
    pb->description.funcs.decode = decode_string;
    pb->description.arg = &_description;
    
}

void OpProto::retrieve(const Nanopb *pb) {

    // name: optional string
    
    // is_commutative: optional bool
    _is_commutative = static_cast<decltype(_is_commutative)>(pb->is_commutative);
    
    // in_num: optional int32
    _in_num = static_cast<decltype(_in_num)>(pb->in_num);
    
    // out_num: optional int32
    _out_num = static_cast<decltype(_out_num)>(pb->out_num);
    
    // description: optional string
    
}

IMPLEMENT_PARSING_WRAPPERS(OpProto);

} // namespace nanopb_cpp
