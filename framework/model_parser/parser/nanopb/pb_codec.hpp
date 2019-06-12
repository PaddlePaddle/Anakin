#ifndef _PB_CODEC_HPP_
#define _PB_CODEC_HPP_

#ifndef _PB_COMMON_HPP_
#error "This header file should not be included alone"
#endif

#include <pb_decode.h>
#include <pb_encode.h>

template<typename T, bool = false>
struct codec_varint {
#ifndef PB_WITHOUT_64BIT
    using delegate_t = uint64_t;
#else
    using delegate_t = uint32_t;
#endif

    static bool decode(pb_istream_t *stream, const pb_field_t *field, void **arg) {
        delegate_t delegate;

        if (!pb_decode_varint(stream, &delegate)) return false;

        *static_cast<T *>(*arg) = static_cast<T>(delegate);

        return true;
    }

    static bool encode(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
        auto src = static_cast<T *>(*arg);
        auto delegate = static_cast<delegate_t>(*src);

        return pb_encode_tag_for_field(stream, field)
            && pb_encode_varint(stream, delegate);
    }
};

template<typename T, bool = false>
struct codec_svarint {
#ifndef PB_WITHOUT_64BIT
    using delegate_t = int64_t;
#else
    using delegate_t = int32_t;
#endif

    static bool decode(pb_istream_t *stream, const pb_field_t *field, void **arg) {
        delegate_t delegate;

        if (!pb_decode_svarint(stream, &delegate)) return false;

        *static_cast<T *>(*arg) = static_cast<T>(delegate);

        return true;
    }

    static bool encode(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
        auto src = static_cast<const T *>(*arg);
        auto delegate = static_cast<const delegate_t>(*src);

        return pb_encode_tag_for_field(stream, field)
            && pb_encode_svarint(stream, delegate);
    }
};

template<typename T, bool = false>
struct codec_fixed32 {
    static bool decode(pb_istream_t *stream, const pb_field_t *field, void **arg) {
        auto dest = static_cast<T *>(*arg);
        auto ret = pb_decode_fixed32(stream, dest);
        return ret;
    }

    static bool encode(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
        return pb_encode_tag_for_field(stream, field)
            && pb_encode_fixed32(stream, static_cast<const T *>(*arg));
    }
};

#ifndef PB_WITHOUT_64BIT
template<typename T, bool = false>
struct codec_fixed64 {
    static bool decode(pb_istream_t *stream, const pb_field_t *field, void **arg) {
        auto dest = static_cast<T *>(*arg);
        return pb_decode_fixed64(stream, dest);
    }

    static bool encode(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
        return pb_encode_tag_for_field(stream, field)
            && pb_encode_fixed64(stream, static_cast<const T *>(*arg));
    }
};
#endif

template<typename T, bool repeat>
struct codec_obj_helper {};

template<typename T>
struct codec_obj_helper<const T, false> {
    static const T *helper(void * const *arg) {
        return static_cast<std::unique_ptr<T> *>(*arg)->get();
    }
};

template<typename T>
struct codec_obj_helper<const T, true> {
    static const T *helper(void * const *arg) {
        return static_cast<const T *>(*arg);
    }
};

template<typename T>
struct codec_obj_helper<T, false> {
    static T *helper(void **arg) {
        return static_cast<std::unique_ptr<T> *>(*arg)->get();
    }
};

template<typename T>
struct codec_obj_helper<T, true> {
    static T *helper(void **arg) {
        return static_cast<T *>(*arg);
    }
};

template<typename T, bool repeat = false>
struct codec_obj {
    static bool decode(pb_istream_t *stream, const pb_field_t *field, void **arg) {
        auto *dest = codec_obj_helper<T, repeat>::helper(arg);
        return dest->ParseFromIstream(stream);
    }

    static bool encode(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
        const auto *src = codec_obj_helper<const T, repeat>::helper(arg);
        typename T::Nanopb submsg = T::NanopbDefault;

        src->pre_encode(&submsg);

        return pb_encode_tag_for_field(stream, field)
            && pb_encode_submessage(stream, T::PBFields, &submsg);
    }
};

template<bool repeat>
struct codec_obj<std::string, repeat> {
    static bool decode(pb_istream_t *stream, const pb_field_t *field, void **arg) {
        auto *str = codec_obj_helper<std::string, repeat>::helper(arg);
        const size_t len = stream->bytes_left;

        str->resize(len);
        std::string::iterator it(str->begin());

        return pb_read(stream, reinterpret_cast<pb_byte_t *>(&*str->begin()), len);
    }

    static bool encode(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
        const auto *str = codec_obj_helper<const std::string, repeat>::helper(arg);

        return pb_encode_tag_for_field(stream, field)
            && pb_encode_string(stream, reinterpret_cast<const pb_byte_t *>(str->data()), str->size());
    }
};

template<typename T, template<typename, bool> class D>
struct codec_repeat {
    static bool decode(pb_istream_t *stream, const pb_field_t *field, void **arg) {
        auto *repeated = static_cast<typename vec_functor<T>::type *>(*arg);
        repeated->push_back(T());
        void *sub_arg = &repeated->back();
        return D<T, true>::decode(stream, field, &sub_arg);
    }

    static bool encode(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
        auto *repeated = static_cast<typename vec_functor<T>::type *>(*arg);

        for (auto &item : *repeated) {
            void *ptr = &item;
            if (!D<T, true>::encode(stream, field, &ptr))
                return false;
        }
        return true;
    }
};

template<typename T>
struct codec_map {
    static bool decode(pb_istream_t *stream, const pb_field_t *field, void **arg) {
        auto *mapping = static_cast<std::map<typename T::KeyType, typename T::ValueType> *>(*arg);
        T map_entry;

        if (!map_entry.ParseFromIstream(stream))
            return false;

        mapping->emplace(std::move(*map_entry.mutable_key()),
                         std::move(*map_entry.mutable_value()));

        return true;
    }

    static bool encode(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
        const auto *mapping = static_cast<const std::map<typename T::KeyType, typename T::ValueType> *>(*arg);

        for (const auto &pair : *mapping) {
            T map_entry(pair.first, pair.second);
            typename T::Nanopb submsg = T::NanopbDefault;

            map_entry.pre_encode(&submsg);

            bool success = pb_encode_tag_for_field(stream, field)
                && pb_encode_submessage(stream, T::PBFields, &submsg);

            if (!success) return false;
        }

        return true;
    }
};

#define IMPLEMENT_CODEC_MEMBERS(MSG_NAME)                               \
    bool MSG_NAME::ParseFromIstream(pb_istream_t *stream) {             \
        Nanopb pb_proto = NanopbDefault;                                \
        pre_decode(&pb_proto);                                          \
        if (!pb_decode(stream, PBFields, &pb_proto))                    \
            return false;                                               \
        post_decode(&pb_proto);                                         \
        return true;                                                    \
    }                                                                   \
    bool MSG_NAME::SerializeToOstream(pb_ostream_t *stream) const {     \
        Nanopb pb_proto;                                                \
        pre_encode(&pb_proto);                                          \
        return pb_encode(stream, PBFields, &pb_proto);                  \
    }

#endif
