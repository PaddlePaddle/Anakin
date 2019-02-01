#ifndef _PB_CPP_COMMON_
#define _PB_CPP_COMMON_

#include <cstdint>
#include <vector>
#include <string>
#include <map>

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

#define REPEATED_PROTO_FIELD(TYPE, NAME)                       \
    PROTO_FIELD(vec_functor<TYPE>::type, NAME)                 \
    const TYPE &NAME(int idx) const {                          \
        auto *ptr = &_##NAME.at(idx);                          \
        return *reinterpret_cast<const TYPE *>(ptr);           \
    }                                                          \
    TYPE *add_##NAME() {                                       \
        _##NAME.push_back(TYPE());                             \
        return reinterpret_cast<TYPE *>(&_##NAME.back());      \
    }                                                          \
    TYPE *add_##NAME(const TYPE &x) {                          \
        _##NAME.push_back(x);                                  \
        return reinterpret_cast<TYPE *>(&_##NAME.back());      \
    }                                                          \
    size_t NAME##_size() const { return _##NAME.size(); }

#define PARSING_MEMBERS(NANOPB_NAME)                                    \
public:                                                                 \
    using Nanopb = ::Nanopb_##NANOPB_NAME;                              \
    static constexpr const pb_field_t *PBFields = NANOPB_NAME##_fields; \
    bool parse_from_buffer(const char *bytes, size_t len);              \
    bool parse_from_file(FILE *f);                                      \
    void fill(Nanopb *p);                                               \
    void retrieve(const Nanopb *p);                                     \
    bool parse(pb_istream_t *stream);

#define PROTO_MAP_ENTRY_MEMBERS(NANOPB_NAME)                            \
public:                                                                 \
    using Nanopb = ::Nanopb_##NANOPB_NAME;                              \
    static constexpr const pb_field_t *PBFields = NANOPB_NAME##_fields; \
    void fill(Nanopb *p);                                               \
    void retrieve(const Nanopb *p);

#define PROTO_MAP_ENTRY_KEY_FIELD(TYPE)   \
public:                                   \
    using KeyType = TYPE;                 \
    PROTO_FIELD(TYPE, key)

#define PROTO_MAP_ENTRY_VALUE_FIELD(TYPE) \
public:                                   \
    using ValueType = TYPE;               \
    PROTO_FIELD(TYPE, value)

#endif // _NANOPB_CPP_COMMON_

