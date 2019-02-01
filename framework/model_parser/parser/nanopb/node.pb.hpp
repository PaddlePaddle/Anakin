#ifndef NANOPB_CPP_NODE_PROTO_HPP
#define NANOPB_CPP_NODE_PROTO_HPP

#include <pb_cpp_common.h>

#include "operator.pb.hpp"
#include "tensor.pb.hpp"

#define valueType Nanopb_valueType
#define NodeProto Nanopb_NodeProto
#define NodeProto_AttrEntry Nanopb_NodeProto_AttrEntry
#define OpProto Nanopb_OpProto
#define TensorShape Nanopb_TensorShape
#define CacheDate Nanopb_CacheDate
#define TensorProto Nanopb_TensorProto
#define TensorShape_Dim Nanopb_TensorShape_Dim
#include "node.pb.h"
#undef valueType
#undef NodeProto
#undef NodeProto_AttrEntry
#undef OpProto
#undef TensorShape
#undef CacheDate
#undef TensorProto
#undef TensorShape_Dim

namespace nanopb_cpp {

class valueType {
    PROTO_FIELD(std::string, s);
    PROTO_FIELD(int32_t, i);
    PROTO_FIELD(float, f);
    PROTO_FIELD(bool, b);
    PROTO_FIELD(nanopb_cpp::CacheDate, cache_list);
    PROTO_FIELD(nanopb_cpp::TensorProto, tensor);
    PROTO_FIELD(nanopb_cpp::DateTypeProto, type);

    PARSING_MEMBERS(valueType);
}; // end class valueType;

class NodeProto {
    class AttrEntry {
        PROTO_MAP_ENTRY_KEY_FIELD(std::string);
        PROTO_MAP_ENTRY_VALUE_FIELD(nanopb_cpp::valueType);

        PROTO_MAP_ENTRY_MEMBERS(NodeProto_AttrEntry);
    }; // end class AttrEntry;

    PROTO_FIELD(std::string, name);
    REPEATED_PROTO_FIELD(std::string, ins);
    REPEATED_PROTO_FIELD(std::string, outs);
    PROTO_FIELD((std::map<std::string, nanopb_cpp::valueType>), attr);
    PROTO_FIELD(int32_t, lane);
    PROTO_FIELD(bool, need_wait);
    PROTO_FIELD(nanopb_cpp::OpProto, op);
    PROTO_FIELD(nanopb_cpp::DateTypeProto, bit_type);

    PARSING_MEMBERS(NodeProto);
}; // end class NodeProto;

} // namespace nanopb_cpp

using nanopb_cpp::valueType;
using nanopb_cpp::NodeProto;


#endif
