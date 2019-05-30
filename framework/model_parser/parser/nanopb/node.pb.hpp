#ifndef NANOPB_CPP_NODE_PROTO_HPP
#define NANOPB_CPP_NODE_PROTO_HPP

#include <pb_common.hpp>

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
#define DateTypeProto Nanopb_DateTypeProto
#include "node.pb.h"
#undef valueType
#undef NodeProto
#undef NodeProto_AttrEntry
#undef OpProto
#undef TensorShape
#undef CacheDate
#undef TensorProto
#undef TensorShape_Dim
#undef DateTypeProto

namespace nanopb_cpp {

class valueType {

    PROTO_SINGULAR_STRING_FIELD(s);

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, i);

    PROTO_SINGULAR_NUMERIC_FIELD(float, f);

    PROTO_SINGULAR_NUMERIC_FIELD(bool, b);

    PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::CacheDate, cache_list);

    PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::TensorProto, tensor);

    PROTO_SINGULAR_ENUM_FIELD(nanopb_cpp::DateTypeProto, type);

    PROTO_MESSAGE_MEMBERS(valueType, valueType);
}; // end class valueType;

class NodeProto {

    class AttrEntry {

        PROTO_SINGULAR_STRING_FIELD(key);

        PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::valueType, value);

    public:
        using KeyType = GET_MEM_TYPE(_key);
        using ValueType = GET_MEM_TYPE(_value);

        AttrEntry(const KeyType &key, const ValueType &value) {
            field_copy(_key, key);
            field_copy(_value, value);
        }

        PROTO_MESSAGE_MEMBERS(AttrEntry, NodeProto_AttrEntry);
    }; // end class AttrEntry;

    PROTO_SINGULAR_STRING_FIELD(name);

    PROTO_REPEATED_STRING_FIELD(ins);

    PROTO_REPEATED_STRING_FIELD(outs);

    PROTO_MAP_FIELD(std::string, nanopb_cpp::valueType, attr);

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, lane);

    PROTO_SINGULAR_NUMERIC_FIELD(bool, need_wait);

    PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::OpProto, op);

    PROTO_SINGULAR_ENUM_FIELD(nanopb_cpp::DateTypeProto, bit_type);

    PROTO_MESSAGE_MEMBERS(NodeProto, NodeProto);
}; // end class NodeProto;

} // namespace nanopb_cpp

using nanopb_cpp::valueType;
using nanopb_cpp::NodeProto;


#endif
