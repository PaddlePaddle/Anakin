#include "adapter.h"

#include <pb_decode.h>

#include <algorithm>
#include <memory>

#ifndef USE_SGX
static size_t fsize(FILE *f) {
    size_t file_len = 0;

    fseek(f, 0, SEEK_END);
    file_len = ftell(f);
    fseek(f, 0, SEEK_SET);

    return file_len;
}
#endif

static pb_istream_t pb_istream_from_file(FILE *f, size_t max_len = SIZE_MAX) {
    size_t file_len = fsize(f);

    auto callback = [](pb_istream_t *stream, pb_byte_t *buf, size_t count) {
        FILE *f = static_cast<FILE *>(stream->state);
        return count == fread(buf, sizeof(pb_byte_t), count, f);
    };

    return {
        .callback = callback,
        .state = f,
        .bytes_left = std::min(file_len, max_len)
    };
}

static bool decode_string(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto str = static_cast<std::string *>(*arg);
    const size_t len = stream->bytes_left;

    str->resize(len);
    std::string::iterator it(str->begin());

    if (!pb_read(stream, reinterpret_cast<pb_byte_t *>(&*str->begin()), len))
        return false;

    return true;
}

template<typename I, typename = typename std::enable_if<std::is_integral<I>::value>::type>
static bool decode_int(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto dest = static_cast<I *>(*arg);
#ifndef PB_WITHOUT_64BIT
    uint64_t delegate;
    if (!pb_decode_varint(stream, &delegate)) return false;
#else
    uint32_t delegate;
    if (!pb_decode_varint32(stream, &delegate)) return false;
#endif

    *dest = static_cast<I>(delegate);
    return true;
}

template<typename I, typename = typename std::enable_if<std::is_signed<I>::value>::type>
static bool decode_sint(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto dest = static_cast<I *>(*arg);
#ifndef PB_WITHOUT_64BIT
    int64_t delegate;
#else
    int32_t delegate;
#endif
    if (!pb_decode_svarint(stream, &delegate)) return false;
    *dest = static_cast<I>(delegate);
    return true;
}

static bool decode_float(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto dest = static_cast<float *>(*arg);
    return pb_decode_fixed32(stream, dest);
}

#ifndef PB_WITHOUT_64BIT
static bool decode_doulbe(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto dest = static_cast<double *>(*arg);
    return pb_decode_fixed64(stream, dest);
}
#endif

template<typename T>
static inline bool decode_proto(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto *dest = static_cast<T *>(*arg);
    return dest->parse(stream);
}

using decoder_t = bool (*)(pb_istream_t *, const pb_field_t *, void **);

template<typename T, decoder_t D>
static bool decode_repeated(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto *repeated = static_cast<std::vector<T> *>(*arg);
    void *sub_arg = nullptr;

    repeated->push_back(T());
    sub_arg = &repeated->back();

    return D(stream, field, &sub_arg);
}

template<typename T>
static bool decode_map(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto *mapping = static_cast<std::map<typename T::KeyType, typename T::ValueType> *>(*arg);

    T adapter_entry;
    typename T::Nanopb pb_entry;

    adapter_entry.fill(&pb_entry);

    if (!pb_decode(stream, T::PBFields, &pb_entry))
        return false;

    adapter_entry.retrieve(&pb_entry);

    mapping->emplace(std::move(*adapter_entry.mutable_key()),
                     std::move(*adapter_entry.mutable_value()));

    return true;
}

namespace anakin {
namespace parser {

#define PROTO_MAP_ENTRY_MEMBERS(PROTO)                                  \
public:                                                                 \
    using Nanopb = ::PROTO;                                             \
    static constexpr const pb_field_t *PBFields = PROTO##_fields;       \
    void fill(Nanopb *p);                                               \
    void retrieve(const Nanopb *p);

#define PROTO_MAP_ENTRY_KEY_FIELD(TYPE) \
    using KeyType = TYPE;               \
    PROTO_FIELD(TYPE, key)

#define PROTO_MAP_ENTRY_VALUE_FIELD(TYPE) \
    using ValueType = TYPE;               \
    PROTO_FIELD(TYPE, value)

class NodeProto_AttrEntry {
    PROTO_MAP_ENTRY_MEMBERS(NodeProto_AttrEntry);

    PROTO_MAP_ENTRY_KEY_FIELD(std::string);
    PROTO_MAP_ENTRY_VALUE_FIELD(valueType);
};

void NodeProto_AttrEntry::fill(Nanopb *pb) {
    pb->key.funcs.decode = decode_string;
    pb->key.arg = &_key;

    _value.fill(&pb->value);
}

void NodeProto_AttrEntry::retrieve(const Nanopb *pb) {
    _value.retrieve(&pb->value);
}

class GraphProto_EdgesInEntry {
    PROTO_MAP_ENTRY_MEMBERS(GraphProto_EdgesInEntry);

    PROTO_MAP_ENTRY_KEY_FIELD(std::string);
    PROTO_MAP_ENTRY_VALUE_FIELD(List);
};

void GraphProto_EdgesInEntry::fill(Nanopb *pb) {
    pb->key.funcs.decode = decode_string;
    pb->key.arg = &_key;

    _value.fill(&pb->value);
}

void GraphProto_EdgesInEntry::retrieve(const Nanopb *pb) {
    _value.retrieve(&pb->value);
}

class GraphProto_EdgesOutEntry {
    PROTO_MAP_ENTRY_MEMBERS(GraphProto_EdgesOutEntry);

    PROTO_MAP_ENTRY_KEY_FIELD(std::string);
    PROTO_MAP_ENTRY_VALUE_FIELD(List);
};

void GraphProto_EdgesOutEntry::fill(Nanopb *pb) {
    pb->key.funcs.decode = decode_string;
    pb->key.arg = &_key;

    _value.fill(&pb->value);
}

void GraphProto_EdgesOutEntry::retrieve(const Nanopb *pb) {
    _value.retrieve(&pb->value);
}

class GraphProto_EdgesInfoEntry {
    PROTO_MAP_ENTRY_MEMBERS(GraphProto_EdgesInfoEntry);

    PROTO_MAP_ENTRY_KEY_FIELD(std::string);
    PROTO_MAP_ENTRY_VALUE_FIELD(TensorProto);
};

void GraphProto_EdgesInfoEntry::fill(Nanopb *pb) {
    pb->key.funcs.decode = decode_string;
    pb->key.arg = &_key;

    _value.fill(&pb->value);
}

void GraphProto_EdgesInfoEntry::retrieve(const Nanopb *pb) {
    _value.retrieve(&pb->value);
}

void OpProto::fill(Nanopb *pb) {
    // name : string
    pb->name.funcs.decode = decode_string;
    pb->name.arg = &_name;

    // description : string
    pb->description.funcs.decode = decode_string;
    pb->description.arg = &_description;
}

void OpProto::retrieve(const Nanopb *pb) {
    _is_commutative = pb->is_commutative;
    _in_num = pb->in_num;
    _out_num = pb->out_num;
}

void CacheDate::fill(Nanopb *pb) {
    // s : repeated string
    pb->s.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->s.arg = &_s;

    // i : repeated int32
    pb->i.funcs.decode = decode_repeated<int32_t, decode_int<int32_t>>;
    pb->i.arg = &_i;

    // f : repeated float
    pb->f.funcs.decode = decode_repeated<float, decode_float>;
    pb->f.arg = &_f;

    /// b : repeated bool
    pb->b.funcs.decode = decode_repeated<unsigned char, decode_int<unsigned char>>;
    pb->b.arg = &_f;

    // l : repeated CacheDate
    pb->l.funcs.decode = decode_repeated<CacheDate, decode_proto<CacheDate>>;
    pb->l.arg = &_l;

    pb->c.funcs.decode = decode_string;
    pb->c.arg = &_c;
}

void CacheDate::retrieve(const Nanopb *pb) {
    _type = static_cast<enum DateTypeProto>(pb->type);
    _size = pb->size;
}

void TensorShape_Dim::fill(Nanopb *pb) {
    ::TensorShape_Dim pb_proto = TensorShape_Dim_init_zero;

    // value : repeated int32
    pb->value.funcs.decode = decode_repeated<int32_t, decode_int<int32_t>>;
    pb->value.arg = &_value;
}

void TensorShape_Dim::retrieve(const Nanopb *pb) {
    _size = pb->size;
}

void TensorShape::fill(Nanopb *pb) {
    // dim : TensorShape_Dim
    _dim.fill(&pb->dim);
}

void TensorShape::retrieve(const Nanopb *pb) {
    _dim.retrieve(&pb->dim);
}

void TensorProto::fill(Nanopb *pb) {
    // name :string
    pb->name.funcs.decode = decode_string;
    pb->name.arg = &_name;

    // share_from : string
    pb->share_from.funcs.decode = decode_string;
    pb->share_from.arg = &_name;

    // shape : TensorShape
    _shape.fill(&pb->shape);

    // valid_shape : TensorShape
    _valid_shape.fill(&pb->valid_shape);

    // data : CacheDate
    _data.fill(&pb->data);

    // scale : CacheDate
    _scale.fill(&pb->scale);
}

void TensorProto::retrieve(const Nanopb *pb) {
    _shared = pb->shared;
    _shape.retrieve(&pb->shape);
    _valid_shape.retrieve(&pb->valid_shape);
    _data.retrieve(&pb->data);
    _scale.retrieve(&pb->scale);
}

// TODO: use union to handle oneof field
void valueType::fill(Nanopb *pb) {
    // s : string
    pb->s.funcs.decode = decode_string;
    pb->s.arg = &_s;

    // cache_list : oneof data { CacheDate }
    _cache_list.fill(&pb->cache_list);

    // cache_list : oneof data { TensorProto }
    _tensor.fill(&pb->tensor);
}

void valueType::retrieve(const Nanopb *pb) {
    _i = pb->i;
    _f = pb->f;
    _b = pb->b;
    _cache_list.retrieve(&pb->cache_list);
    _tensor.retrieve(&pb->tensor);
    _type = static_cast<enum DateTypeProto>(pb->type);
}

void NodeProto::fill(Nanopb *pb) {
    // name : string
    pb->name.funcs.decode = decode_string;
    pb->name.arg = &_name;

    // ins : repeated string
    pb->ins.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->ins.arg = &_ins;

    // outs : repeated string
    pb->outs.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->outs.arg = &_ins;

    // attr : map<string, valueType>
    pb->attr.funcs.decode = decode_map<NodeProto_AttrEntry>;
    pb->attr.arg = &_attr;

    // Op : OpProto
    _op.fill(&pb->Op);
}

void NodeProto::retrieve(const Nanopb *pb) {
    _lane = pb->lane;
    _need_wait = pb->need_wait;
    _bit_type = static_cast<enum DateTypeProto>(pb->bit_type);

    _op.retrieve(&pb->Op);
}

void Version::fill(Nanopb *pb) {}

void Version::retrieve(const Nanopb *pb) {
    _major = pb->major;
    _minor = pb->minor;
    _patch = pb->patch;
    _version = pb->version;
}

void Info::fill(Nanopb *pb) {}

void Info::retrieve(const Nanopb *pb) {
    _temp_mem_used = pb->temp_mem_used;
    _original_temp_mem_used = pb->original_temp_mem_used;
    _system_mem_used = pb->system_mem_used;
    _is_optimized = pb->is_optimized;
}

void TargetProto::fill(Nanopb *pb) {
    // node : string
    pb->node.funcs.decode = decode_string;
    pb->node.arg = &_node;

    pb->scale.funcs.decode = decode_repeated<float, decode_float>;
    pb->scale.arg = &_scale;
}

void TargetProto::retrieve(const Nanopb *pb) {}

void List::fill(Nanopb *pb) {
    // val : repeated string
    pb->val.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->val.arg = &_val;

    // target : repeated TargetProto
    pb->target.funcs.decode = decode_repeated<TargetProto, decode_proto<TargetProto>>;
    pb->target.arg = &_target;
}

void List::retrieve(const Nanopb *pb) {}

void GraphProto::fill(Nanopb *pb) {
    // name : string
    pb->name.funcs.decode = decode_string;
    pb->name.arg = &_name;

    // nodes : repeated NodeProto
    pb->nodes.funcs.decode = decode_repeated<NodeProto, decode_proto<NodeProto>>;
    pb->nodes.arg = &_nodes;

    // edges_in : map<string, List>
    pb->edges_in.funcs.decode = decode_map<GraphProto_EdgesInEntry>;
    pb->edges_in.arg = &_edges_in;

    // edges_out : map<string, List>
    pb->edges_out.funcs.decode = decode_map<GraphProto_EdgesOutEntry>;
    pb->edges_out.arg = &_edges_out;

    // edges_info : map<string, TensorProto>
    pb->edges_info.funcs.decode = decode_map<GraphProto_EdgesInfoEntry>;
    pb->edges_info.arg = &_edges_info;

    // ins : repeated string
    pb->ins.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->ins.arg = &_ins;

    // outs : repeated string
    pb->outs.funcs.decode = decode_repeated<std::string, decode_string>;
    pb->outs.arg = &_outs;

    // summary : Info
    _summary.fill(&pb->summary);
}

void GraphProto::retrieve(const Nanopb *pb) {
    _summary.retrieve(&pb->summary);
}

#define IMPLEMENT_PARSING_WRAPPERS(PROTO)                               \
    bool PROTO::parse_from_file(FILE *f) {                              \
        auto stream = pb_istream_from_file(f);                          \
        return parse(&stream);                                          \
    }                                                                   \
    bool PROTO::parse_from_buffer(const char *buffer, size_t len) {     \
        auto stream = pb_istream_from_buffer(                           \
            reinterpret_cast<const pb_byte_t *>(buffer), len);          \
        return parse(&stream);                                          \
    }                                                                   \
    bool PROTO::parse(pb_istream_t *stream) {                           \
        Nanopb pb_proto;                                                \
        fill(&pb_proto);                                                \
        if (!pb_decode(stream, PROTO##_fields, &pb_proto))              \
            return false;                                               \
        retrieve(&pb_proto);                                            \
        return true;                                                    \
    }

IMPLEMENT_PARSING_WRAPPERS(OpProto);
IMPLEMENT_PARSING_WRAPPERS(CacheDate);
IMPLEMENT_PARSING_WRAPPERS(TensorShape_Dim);
IMPLEMENT_PARSING_WRAPPERS(TensorShape);
IMPLEMENT_PARSING_WRAPPERS(TensorProto);
IMPLEMENT_PARSING_WRAPPERS(valueType);
IMPLEMENT_PARSING_WRAPPERS(NodeProto);
IMPLEMENT_PARSING_WRAPPERS(Version);
IMPLEMENT_PARSING_WRAPPERS(Info);
IMPLEMENT_PARSING_WRAPPERS(TargetProto);
IMPLEMENT_PARSING_WRAPPERS(List);
IMPLEMENT_PARSING_WRAPPERS(GraphProto);

}
}
