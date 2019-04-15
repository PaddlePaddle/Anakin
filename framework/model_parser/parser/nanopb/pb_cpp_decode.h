#ifndef NANOPB_DECODE_CPP_H
#define NANOPB_DECODE_CPP_H

#include <pb_decode.h>

#include <algorithm>
#include <memory>
#include <map>
#include <string>

#include <stdio.h>

#include "anakin_config.h"

template<typename I>
static bool decode_varint(pb_istream_t *stream, const pb_field_t *field, void **arg) {
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

template<typename I>
static bool decode_svarint(pb_istream_t *stream, const pb_field_t *field, void **arg) {
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

template<typename T>
bool decode_fixed32(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto dest = static_cast<T *>(*arg);
    auto ret = pb_decode_fixed32(stream, dest);
    return ret;
}

#ifndef PB_WITHOUT_64BIT
template<typename T>
bool decode_fixed64(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto dest = static_cast<T *>(*arg);
    return pb_decode_fixed64(stream, dest);
}
#endif

template<typename T>
bool decode_message(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto *dest = static_cast<T *>(*arg);
    return dest->parse(stream);
}

using decoder_t = bool (*)(pb_istream_t *, const pb_field_t *, void **);

template<typename T, decoder_t D>
bool decode_repeated(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto *repeated = static_cast<typename vec_functor<T>::type *>(*arg);
    repeated->push_back(T());
    void *sub_arg = &repeated->back();
    return D(stream, field, &sub_arg);
}

template<typename T>
bool decode_map(pb_istream_t *stream, const pb_field_t *field, void **arg) {
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

template<typename = std::string>
bool decode_string(pb_istream_t *stream, const pb_field_t *field, void **arg) {
    auto str = static_cast<std::string *>(*arg);
    const size_t len = stream->bytes_left;
    str->resize(len);
    std::string::iterator it(str->begin());
    if (!pb_read(stream, reinterpret_cast<pb_byte_t *>(&*str->begin()), len))
        return false;
    return true;
}

static size_t file_size(FILE *f) {
    size_t file_len;

    fseek(f, 0, SEEK_END);
    file_len = ftell(f);
    fseek(f, 0, SEEK_SET);

    return file_len;
}

#define IMPLEMENT_PARSING_WRAPPERS(PROTO)                               \
    bool PROTO::parse_from_file(FILE *f) {                              \
        size_t file_len = file_size(f);                                 \
        auto callback = [](pb_istream_t *stream, pb_byte_t *buf,        \
                           size_t count) {                              \
            FILE *f = static_cast<FILE *>(stream->state);               \
            return count == fread(buf, sizeof(pb_byte_t), count, f);    \
        };                                                              \
        pb_istream_t stream {                                           \
            .callback = callback,                                       \
            .state = f,                                                 \
            .bytes_left = file_len,                                     \
        };                                                              \
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
        if (!pb_decode(stream, PBFields, &pb_proto))                    \
            return false;                                               \
        retrieve(&pb_proto);                                            \
        return true;                                                    \
    }

#endif
