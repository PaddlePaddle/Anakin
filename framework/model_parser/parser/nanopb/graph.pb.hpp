#ifndef NANOPB_CPP_GRAPH_PROTO_HPP
#define NANOPB_CPP_GRAPH_PROTO_HPP

#include <pb_common.hpp>

#include "node.pb.hpp"
#include "tensor.pb.hpp"

#define Version Nanopb_Version
#define Info Nanopb_Info
#define TargetProto Nanopb_TargetProto
#define List Nanopb_List
#define GraphProto Nanopb_GraphProto
#define GraphProto_EdgesInEntry Nanopb_GraphProto_EdgesInEntry
#define GraphProto_EdgesOutEntry Nanopb_GraphProto_EdgesOutEntry
#define GraphProto_EdgesInfoEntry Nanopb_GraphProto_EdgesInfoEntry
#define valueType Nanopb_valueType
#define NodeProto Nanopb_NodeProto
#define NodeProto_AttrEntry Nanopb_NodeProto_AttrEntry
#define TensorShape Nanopb_TensorShape
#define CacheDate Nanopb_CacheDate
#define TensorProto Nanopb_TensorProto
#define TensorShape_Dim Nanopb_TensorShape_Dim
#define LayoutProto Nanopb_LayoutProto
#define DateTypeProto Nanopb_DateTypeProto
#include "graph.pb.h"
#undef Version
#undef Info
#undef TargetProto
#undef List
#undef GraphProto
#undef GraphProto_EdgesInEntry
#undef GraphProto_EdgesOutEntry
#undef GraphProto_EdgesInfoEntry
#undef valueType
#undef NodeProto
#undef NodeProto_AttrEntry
#undef TensorShape
#undef CacheDate
#undef TensorProto
#undef TensorShape_Dim
#undef LayoutProto
#undef DateTypeProto

namespace nanopb_cpp {

enum LayoutProto {
    Invalid = 0,
    LP_W = 1,
    LP_HW = 2,
    LP_WH = 3,
    LP_NC = 4,
    LP_NH = 5,
    LP_NW = 6,
    LP_NHW = 7,
    LP_NCHW = 8,
    LP_NHWC = 9,
    LP_NCHW_C4 = 10,
    LP_NCHW_C8 = 11,
    LP_NCHW_C16 = 12,
    LP_OIHW16I16O = 13,
    LP_GOIHW16I16O = 14,
    LP_NCHW_C8R = 15,
    LP_NCHW_C16R = 16,
};

class Version {

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, major);

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, minor);

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, patch);

    PROTO_SINGULAR_NUMERIC_FIELD(int64_t, version);

    PROTO_MESSAGE_MEMBERS(Version, Version);
}; // end class Version;

class Info {

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, temp_mem_used);

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, original_temp_mem_used);

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, system_mem_used);

    PROTO_SINGULAR_NUMERIC_FIELD(int32_t, model_mem_used);

    PROTO_SINGULAR_NUMERIC_FIELD(bool, is_optimized);

    PROTO_MESSAGE_MEMBERS(Info, Info);
}; // end class Info;

class TargetProto {

    PROTO_SINGULAR_STRING_FIELD(node);

    PROTO_REPEATED_NUMERIC_FIELD(float, scale);

    PROTO_SINGULAR_ENUM_FIELD(nanopb_cpp::LayoutProto, layout);

    PROTO_MESSAGE_MEMBERS(TargetProto, TargetProto);
}; // end class TargetProto;

class List {

    PROTO_REPEATED_STRING_FIELD(val);

    PROTO_REPEATED_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::TargetProto, target);

    PROTO_MESSAGE_MEMBERS(List, List);
}; // end class List;

class GraphProto {

    class EdgesInEntry {

        PROTO_SINGULAR_STRING_FIELD(key);

        PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::List, value);

    public:
        using KeyType = GET_MEM_TYPE(_key);
        using ValueType = GET_MEM_TYPE(_value);

        EdgesInEntry(const KeyType &key, const ValueType &value) {
            field_copy(_key, key);
            field_copy(_value, value);
        }

        PROTO_MESSAGE_MEMBERS(EdgesInEntry, GraphProto_EdgesInEntry);
    }; // end class EdgesInEntry;

    class EdgesOutEntry {

        PROTO_SINGULAR_STRING_FIELD(key);

        PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::List, value);

    public:
        using KeyType = GET_MEM_TYPE(_key);
        using ValueType = GET_MEM_TYPE(_value);

        EdgesOutEntry(const KeyType &key, const ValueType &value) {
            field_copy(_key, key);
            field_copy(_value, value);
        }

        PROTO_MESSAGE_MEMBERS(EdgesOutEntry, GraphProto_EdgesOutEntry);
    }; // end class EdgesOutEntry;

    class EdgesInfoEntry {

        PROTO_SINGULAR_STRING_FIELD(key);

        PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::TensorProto, value);

    public:
        using KeyType = GET_MEM_TYPE(_key);
        using ValueType = GET_MEM_TYPE(_value);

        EdgesInfoEntry(const KeyType &key, const ValueType &value) {
            field_copy(_key, key);
            field_copy(_value, value);
        }

        PROTO_MESSAGE_MEMBERS(EdgesInfoEntry, GraphProto_EdgesInfoEntry);
    }; // end class EdgesInfoEntry;

    PROTO_SINGULAR_STRING_FIELD(name);

    PROTO_REPEATED_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::NodeProto, nodes);

    PROTO_MAP_FIELD(std::string, nanopb_cpp::List, edges_in);

    PROTO_MAP_FIELD(std::string, nanopb_cpp::List, edges_out);

    PROTO_MAP_FIELD(std::string, nanopb_cpp::TensorProto, edges_info);

    PROTO_REPEATED_STRING_FIELD(ins);

    PROTO_REPEATED_STRING_FIELD(outs);

    PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::Version, version);

    PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(nanopb_cpp::Info, summary);

    PROTO_MESSAGE_MEMBERS(GraphProto, GraphProto);
}; // end class GraphProto;

} // namespace nanopb_cpp

using nanopb_cpp::Version;
using nanopb_cpp::Info;
using nanopb_cpp::TargetProto;
using nanopb_cpp::List;
using nanopb_cpp::GraphProto;

using nanopb_cpp::LayoutProto;
using nanopb_cpp::Invalid;
using nanopb_cpp::LP_W;
using nanopb_cpp::LP_HW;
using nanopb_cpp::LP_WH;
using nanopb_cpp::LP_NC;
using nanopb_cpp::LP_NH;
using nanopb_cpp::LP_NW;
using nanopb_cpp::LP_NHW;
using nanopb_cpp::LP_NCHW;
using nanopb_cpp::LP_NHWC;
using nanopb_cpp::LP_NCHW_C4;
using nanopb_cpp::LP_NCHW_C8;
using nanopb_cpp::LP_NCHW_C16;
using nanopb_cpp::LP_OIHW16I16O;
using nanopb_cpp::LP_GOIHW16I16O;
using nanopb_cpp::LP_NCHW_C8R;
using nanopb_cpp::LP_NCHW_C16R;

#endif
