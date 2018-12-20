// -*- c++ -*-
#ifndef ANAKIN_NANOPB_ADAPTER_H
#define ANAKIN_NANOPB_ADAPTER_H

#include "anakin_config.h"

#include "operator.pb.h"
#include "tensor.pb.h"
#include "node.pb.h"
#include "graph.pb.h"

#include <vector>
#include <string>
#include <map>
#include <functional>
#include "stdio.h"

namespace anakin {
namespace parser {
// define the C++ message types in anakin::parser to override the
// corresponding C types defined by nanopb in the gloabl namespace

template<size_t S> struct bool_adaptor {};
template<> struct bool_adaptor<1> { using type = uint8_t; };
template<> struct bool_adaptor<2> { using type = uint16_t; };
template<> struct bool_adaptor<4> { using type = uint32_t; };
template<> struct bool_adaptor<8> { using type = uint64_t; };

template<typename T>
struct vec_functor {
    using type = std::vector<T>;
};
template<>
struct vec_functor<bool> {
    using type = std::vector<typename bool_adaptor<sizeof(bool)>::type>;
};

template<typename T> struct argument_type {};
template<typename U, typename T> struct argument_type<U(T)> {
    using type = T;
};

#define PROTO_TY(TYPE) typename argument_type<void(TYPE)>::type

#define PROTO_FIELD(TYPE, NAME)                                \
private:                                                       \
    PROTO_TY(TYPE) _##NAME;                                    \
public:                                                        \
    PROTO_TY(TYPE) *mutable_##NAME() { return &_##NAME; }      \
    void set_##NAME(const PROTO_TY(TYPE) &x) { _##NAME = x; }  \
    const PROTO_TY(TYPE) &NAME() const { return _##NAME; }

#define REPEATED_PROTO_FIELD(TYPE, NAME)                        \
    PROTO_FIELD(vec_functor<TYPE>::type, NAME)                  \
    TYPE *add_##NAME() {                                        \
        _##NAME.push_back(TYPE());                              \
        return reinterpret_cast<TYPE *>(&_##NAME.back());       \
    }                                                           \
    TYPE *add_##NAME(const TYPE &x) {                           \
        _##NAME.push_back(x);                                   \
        return reinterpret_cast<TYPE *>(&_##NAME.back());       \
    }

#define PARSING_MEMBERS(PROTO)                              \
public:                                                     \
    using Nanopb = ::PROTO;                                 \
    bool parse_from_buffer(const char *bytes, size_t len);  \
    bool parse_from_file(FILE *f);                          \
    void fill(Nanopb *p);                                   \
    void retrieve(const Nanopb *p);                         \
    bool parse(pb_istream_t *stream);

enum DateTypeProto {
    STR        =  DateTypeProto_STR,
    INT8       =  DateTypeProto_INT8,
    INT32      =  DateTypeProto_INT32,
    FLOAT      =  DateTypeProto_FLOAT,
    DOUBLE     =  DateTypeProto_DOUBLE,
    BOOLEN     =  DateTypeProto_BOOLEN,
    CACHE_LIST =  DateTypeProto_CACHE_LIST,
    TENSOR     =  DateTypeProto_TENSOR,
};

class OpProto {
    PROTO_FIELD(std::string, name);
    PROTO_FIELD(bool, is_commutative);
    PROTO_FIELD(int32_t, in_num);
    PROTO_FIELD(int32_t, out_num);
    PROTO_FIELD(std::string, description);

    PARSING_MEMBERS(OpProto);
};

class CacheDate {
    REPEATED_PROTO_FIELD(std::string, s);
    REPEATED_PROTO_FIELD(int32_t, i);
    REPEATED_PROTO_FIELD(float, f);
    REPEATED_PROTO_FIELD(bool, b);
    REPEATED_PROTO_FIELD(CacheDate, l);
    PROTO_FIELD(std::string, c);
    PROTO_FIELD(DateTypeProto, type);
    PROTO_FIELD(int64_t, size);

    PARSING_MEMBERS(CacheDate);
};

class TensorShape_Dim {
    REPEATED_PROTO_FIELD(int32_t, value);
    PROTO_FIELD(int64_t, size);

    PARSING_MEMBERS(TensorShape_Dim);
};

class TensorShape {
    PROTO_FIELD(TensorShape_Dim, dim);

    PARSING_MEMBERS(TensorShape);
};

class TensorProto {
    PROTO_FIELD(std::string, name);
    PROTO_FIELD(bool, shared);
    PROTO_FIELD(std::string, share_from);
    PROTO_FIELD(TensorShape, shape);
    PROTO_FIELD(TensorShape, valid_shape);
    PROTO_FIELD(CacheDate, data);
    PROTO_FIELD(CacheDate, scale);

    PARSING_MEMBERS(TensorProto);
};

class valueType {
    PROTO_FIELD(std::string, s);
    PROTO_FIELD(int32_t, i);
    PROTO_FIELD(float, f);
    PROTO_FIELD(bool, b);
    PROTO_FIELD(CacheDate, cache_list);
    PROTO_FIELD(TensorProto, tensor);
    PROTO_FIELD(DateTypeProto, type);

    PARSING_MEMBERS(valueType);
};

class NodeProto {
    PROTO_FIELD(std::string, name);
    REPEATED_PROTO_FIELD(std::string, ins);
    REPEATED_PROTO_FIELD(std::string, outs);
    PROTO_FIELD((std::map<std::string, valueType>), attr);
    PROTO_FIELD(int32_t, lane);
    PROTO_FIELD(bool, need_wait);
    PROTO_FIELD(OpProto, op);
    PROTO_FIELD(DateTypeProto, bit_type);

    PARSING_MEMBERS(NodeProto);
};

class Version {
    PROTO_FIELD(int32_t, major);
    PROTO_FIELD(int32_t, minor);
    PROTO_FIELD(int32_t, patch);
    PROTO_FIELD(int64_t, version);

    PARSING_MEMBERS(Version);
};

class Info {
    PROTO_FIELD(int32_t, temp_mem_used);
    PROTO_FIELD(int32_t, original_temp_mem_used);
    PROTO_FIELD(int32_t, system_mem_used);
    PROTO_FIELD(int32_t, model_mem_used);
    PROTO_FIELD(bool, is_optimized);

    PARSING_MEMBERS(Info);
};

class TargetProto {
    PROTO_FIELD(std::string, node);
    REPEATED_PROTO_FIELD(float, scale);

    PARSING_MEMBERS(TargetProto);
};

class List {
    REPEATED_PROTO_FIELD(std::string, val);
    REPEATED_PROTO_FIELD(TargetProto, target);

    PARSING_MEMBERS(List);
};

class GraphProto {
    PROTO_FIELD(std::string, name);
    REPEATED_PROTO_FIELD(NodeProto, nodes);
    PROTO_FIELD((std::map<std::string, List>), edges_in);
    PROTO_FIELD((std::map<std::string, List>), edges_out);
    PROTO_FIELD((std::map<std::string, TensorProto>), edges_info);
    REPEATED_PROTO_FIELD(std::string, ins);
    REPEATED_PROTO_FIELD(std::string, outs);
    PROTO_FIELD(Version, version);
    PROTO_FIELD(Info, summary);

    PARSING_MEMBERS(GraphProto);
};

} // parser
} // anakin

#endif
