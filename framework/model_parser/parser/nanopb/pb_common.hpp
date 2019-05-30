#ifndef _PB_COMMON_HPP_
#define _PB_COMMON_HPP_

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <pb_decode.h>
#include <pb_encode.h>

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

#define PROTO_SINGULAR_NUMERIC_FIELD(TYPE, NAME)                \
private:                                                        \
    TYPE _##NAME = static_cast<TYPE>(0);                        \
public:                                                         \
    TYPE NAME() const { return _##NAME; }                       \
    void set_##NAME(TYPE value) { _##NAME = value; }            \
    void clear_##NAME() { _##NAME = static_cast<TYPE>(0); }

#define PROTO_SINGULAR_STRING_FIELD(NAME)                       \
private:                                                        \
    std::unique_ptr<std::string> _##NAME =                      \
        std::unique_ptr<std::string>(new std::string());        \
public:                                                         \
    const std::string &NAME() const { return *_##NAME; }        \
    std::string *mutable_##NAME() { return _##NAME.get(); }     \
    void set_##NAME(const std::string &value) {                 \
        *_##NAME = value;                                       \
    }                                                           \
    void set_##NAME(std::string &&value) {                      \
        *_##NAME = std::move(value);                            \
    }                                                           \
    void set_##NAME(const char *value, int size) {              \
        _##NAME->resize(size);                                  \
        std::copy(value, value + size, _##NAME->begin());       \
    }                                                           \
    void set_##NAME(const char *value) {                        \
        set_##NAME(value, std::strlen(value));                  \
    }                                                           \
    void clear_##NAME() { _##NAME->clear(); }                   \
    void set_allocated_##NAME(std::string *value) {             \
        if (value) {                                            \
            _##NAME.reset(value);                               \
        } else {                                                \
            clear_##NAME();                                     \
        }                                                       \
    }                                                           \
    std::string *release_##NAME() { return _##NAME.release(); }

#define PROTO_SINGULAR_ENUM_FIELD(TYPE, NAME)                   \
private:                                                        \
    TYPE _##NAME = static_cast<TYPE>(0);                        \
public:                                                         \
    TYPE NAME() const { return _##NAME; }                       \
    void set_##NAME(TYPE value) { _##NAME = value; }            \
    void clear_foo() { _##NAME = static_cast<TYPE>(0); }

#define PROTO_SINGULAR_EMBEDDED_MESSAGE_FIELD(TYPE, NAME)               \
private:                                                                \
    std::unique_ptr<TYPE> _##NAME =                                     \
        std::unique_ptr<TYPE>(new TYPE());                              \
public:                                                                 \
    bool has_##NAME() const { return (bool)_##NAME; }                   \
    const TYPE &NAME () const {                                         \
        if (has_##NAME()) return *_##NAME;                              \
        else return TYPE::default_instance();                           \
    }                                                                   \
    TYPE *mutable_##NAME() { return _##NAME.get(); }                    \
    void clear_##NAME() { _##NAME.reset(new TYPE()); }                  \
    void set_allocated_##NAME(TYPE *value) {_##NAME.reset(value); }     \
    TYPE *release_##NAME() { return _##NAME.release(); }

#define PROTO_REPEATED_NUMERIC_FIELD(TYPE, NAME)                          \
private:                                                                  \
    vec_functor<TYPE>::type _##NAME;                                      \
public:                                                                   \
    int NAME##_size() const { return _##NAME.size(); }                    \
    TYPE NAME(int index) const { return _##NAME.at(index); }              \
    void set_##NAME(int index, TYPE value) { _##NAME.at(index) = value; } \
    void add_##NAME(TYPE value) { _##NAME.push_back(value); }             \
    void clear_##NAME() { _##NAME.clear(); }                              \
    const vec_functor<TYPE>::type &NAME() const { return _##NAME; }       \
    vec_functor<TYPE>::type *mutable_##NAME() { return &_##NAME; }

#define PROTO_REPEATED_STRING_FIELD(NAME)                               \
private:                                                                \
    std::vector<std::string> _##NAME;                                   \
public:                                                                 \
    int NAME##_size() const { return _##NAME.size(); }                  \
    const std::string &NAME(int index) const {                          \
        return _##NAME.at(index);                                       \
    }                                                                   \
    void set_##NAME(int index, std::string value) {                     \
        _##NAME.at(index) = std::move(value);                           \
    }                                                                   \
    void set_##NAME(int index, const char * value, int len) {           \
        auto &str = _##NAME.at(index);                                  \
        str.resize(len);                                                \
        std::copy(value, value + len, str.begin());                     \
    }                                                                   \
    void set_##NAME(int index, const char *value) {                     \
        auto len = std::strlen(value);                                  \
        set_##NAME(index, value, len);                                  \
    }                                                                   \
    void add_##NAME(std::string value) {                                \
        _##NAME.push_back(std::move(value));                            \
    }                                                                   \
    void add_##NAME(const char *value) {                                \
        _##NAME.emplace_back(value);                                    \
    }                                                                   \
    void add_##NAME(const char *value, int len) {                       \
        _##NAME.emplace_back(value, len);                               \
    }                                                                   \
    void clear_##NAME() { _##NAME.clear(); }                            \
    const std::vector<std::string> &NAME() const {                      \
        return _##NAME;                                                 \
    }                                                                   \
    std::vector<std::string> *mutable_##NAME() { return &_##NAME; }

#define PROTO_REPEATED_ENUM_FIELD(TYPE, NAME)                             \
private:                                                                  \
    std::vector<TYPE> _##NAME;                                            \
public:                                                                   \
    int NAME##_size() const { return _##NAME.size(); }                    \
    TYPE NAME(int index) const { return _##NAME.at(index); }              \
    void set_##NAME(int index, TYPE value) { _##NAME.at(index) = value; } \
    void add_##NAME(TYPE value) { _##NAME.push_back(value); }             \
    void clear_##NAME() { _##NAME.clear(); }                              \
    const std::vector<TYPE> &NAME() const { return _##NAME; }             \
    std::vector<TYPE> *mutable_##NAME() { return &_##NAME; }

#define PROTO_REPEATED_EMBEDDED_MESSAGE_FIELD(TYPE, NAME)               \
private:                                                                \
    std::vector<TYPE> _##NAME;                                          \
public:                                                                 \
    int NAME##_size() const { return _##NAME.size(); }                  \
    const TYPE &NAME(int index) const { return _##NAME.at(index); }     \
    TYPE *mutable_##NAME(int index) { return &_##NAME.at(index); }      \
    TYPE *add_##NAME() {                                                \
        _##NAME.emplace_back();                                         \
        return &_##NAME.back();                                         \
    }                                                                   \
    void clear_##NAME() { _##NAME.clear(); }                            \
    const std::vector<TYPE> &NAME() const { return _##NAME; }           \
    std::vector<TYPE> *mutable_##NAME() { return &_##NAME; }

#define PROTO_MAP_FIELD(KEY_TYPE, FIELD_TYPE, NAME)                        \
private:                                                                   \
    std::map<KEY_TYPE, FIELD_TYPE> _##NAME;                                \
public:                                                                    \
    const std::map<KEY_TYPE, FIELD_TYPE> &NAME() const { return _##NAME; } \
    std::map<KEY_TYPE, FIELD_TYPE> *mutable_##NAME() { return &_##NAME; }

#define PROTO_MESSAGE_MEMBERS(MSG_NAME, NANOPB_NAME)                          \
public:                                                                       \
    using Nanopb = ::Nanopb_##NANOPB_NAME;                                    \
    static constexpr const pb_field_t *PBFields = NANOPB_NAME##_fields;       \
    static constexpr const Nanopb NanopbDefault = NANOPB_NAME##_init_default; \
    MSG_NAME() {}                                                             \
    MSG_NAME(const MSG_NAME &other) { this->CopyFrom(other); }                \
    MSG_NAME &operator=(const MSG_NAME &other) {                        \
        this->CopyFrom(other);                                          \
        return *this;                                                   \
    }                                                                   \
    void pre_decode(Nanopb *p);                                               \
    void post_decode(const Nanopb *p);                                        \
    void pre_encode(Nanopb *p) const;                                         \
    void CopyFrom(const MSG_NAME&);                                           \
    static const MSG_NAME &default_instance() {                               \
        static MSG_NAME _default_instance{};                                  \
        return _default_instance;                                             \
    }                                                                         \
    bool ParseFromIstream(pb_istream_t *stream);                              \
    bool SerializeToOstream(pb_ostream_t *stream) const;

template<typename T>
inline void field_copy(std::unique_ptr<T> &to, const std::unique_ptr<T> &from) {
    to.reset(new T(*from));
}

template<typename T>
inline void field_copy(std::unique_ptr<T> &to, const T &from) {
    to.reset(new T(from));
}

template<typename T>
inline void field_copy(T &to, const T &from) {
    to = from;
}

template<typename M> M get_member_type(std::unique_ptr<M> *);
template<typename M> M get_member_type(M *);

#define GET_MEM_TYPE(mem) decltype(get_member_type(&mem))

#endif // _NANOPB_CPP_COMMON_
