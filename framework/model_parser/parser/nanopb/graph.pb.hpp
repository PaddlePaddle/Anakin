#ifndef NANOPB_CPP_GRAPH_PROTO_HPP
#define NANOPB_CPP_GRAPH_PROTO_HPP

#include <pb_cpp_common.h>

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
    PROTO_FIELD(int32_t, major);
    PROTO_FIELD(int32_t, minor);
    PROTO_FIELD(int32_t, patch);
    PROTO_FIELD(int64_t, version);

    PARSING_MEMBERS(Version);
}; // end class Version;

class Info {
    PROTO_FIELD(int32_t, temp_mem_used);
    PROTO_FIELD(int32_t, original_temp_mem_used);
    PROTO_FIELD(int32_t, system_mem_used);
    PROTO_FIELD(int32_t, model_mem_used);
    PROTO_FIELD(bool, is_optimized);

    PARSING_MEMBERS(Info);
}; // end class Info;

class TargetProto {
    PROTO_FIELD(std::string, node);
    REPEATED_PROTO_FIELD(float, scale);
    PROTO_FIELD(nanopb_cpp::LayoutProto, layout);

    PARSING_MEMBERS(TargetProto);
}; // end class TargetProto;

class List {
    REPEATED_PROTO_FIELD(std::string, val);
    REPEATED_PROTO_FIELD(nanopb_cpp::TargetProto, target);

    PARSING_MEMBERS(List);
}; // end class List;

class GraphProto {
    class EdgesInEntry {
        PROTO_MAP_ENTRY_KEY_FIELD(std::string);
        PROTO_MAP_ENTRY_VALUE_FIELD(nanopb_cpp::List);

        PROTO_MAP_ENTRY_MEMBERS(GraphProto_EdgesInEntry);
    }; // end class EdgesInEntry;

    class EdgesOutEntry {
        PROTO_MAP_ENTRY_KEY_FIELD(std::string);
        PROTO_MAP_ENTRY_VALUE_FIELD(nanopb_cpp::List);

        PROTO_MAP_ENTRY_MEMBERS(GraphProto_EdgesOutEntry);
    }; // end class EdgesOutEntry;

    class EdgesInfoEntry {
        PROTO_MAP_ENTRY_KEY_FIELD(std::string);
        PROTO_MAP_ENTRY_VALUE_FIELD(nanopb_cpp::TensorProto);

        PROTO_MAP_ENTRY_MEMBERS(GraphProto_EdgesInfoEntry);
    }; // end class EdgesInfoEntry;

    PROTO_FIELD(std::string, name);
    REPEATED_PROTO_FIELD(nanopb_cpp::NodeProto, nodes);
    PROTO_FIELD((std::map<std::string, nanopb_cpp::List>), edges_in);
    PROTO_FIELD((std::map<std::string, nanopb_cpp::List>), edges_out);
    PROTO_FIELD((std::map<std::string, nanopb_cpp::TensorProto>), edges_info);
    REPEATED_PROTO_FIELD(std::string, ins);
    REPEATED_PROTO_FIELD(std::string, outs);
    PROTO_FIELD(nanopb_cpp::Version, version);
    PROTO_FIELD(nanopb_cpp::Info, summary);

    PARSING_MEMBERS(GraphProto);
}; // end class GraphProto;

} // namespace nanopb_cpp

using nanopb_cpp::Version;
using nanopb_cpp::Info;
using nanopb_cpp::TargetProto;
using nanopb_cpp::List;
using nanopb_cpp::GraphProto;

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
