#ifndef ANAKIN_SABER_FUNCS_IMPL_SABER_UTIL_H
#define ANAKIN_SABER_FUNCS_IMPL_SABER_UTIL_H
#include <assert.h>
#include "saber/core/common.h"
#include "saber/core/tensor.h"
#include "saber/core/shape.h"
namespace anakin {

namespace saber {
namespace utils {


template<typename opTensor>
static inline bool try_expand_tensor(opTensor& x, anakin::saber::Shape shape) {
    if (x.valid_size() < shape.count()) {
        x.re_alloc(shape, x.get_dtype());
        return true;
    }

    return false;
}

template<typename opTensor>
static inline bool try_expand_tensor(opTensor& x, int size) {
    if (x.valid_size() < size) {
        anakin::saber::Shape shape({1, 1, 1, size}, Layout_NCHW);
        return try_expand_tensor(x, shape);
    }

    return false;
}

template <typename DataType>
static inline void transpose(const DataType* in, int height, int width, DataType* out) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            out[j * height + i] = in[i * width + j];
        }
    }
}

inline int round_up(int k, int c) {
    return ((k + c - 1) / c) * c;
}

inline int div_up(int k, int c) {
    return (k + c - 1) / c;
}


/* a bunch of std:: analogues to be compliant with any msvs version
 *
 * Rationale: msvs c++ (and even some c) headers contain special pragma that
 * injects msvs-version check into object files in order to abi-mismatches
 * during the static linking. This makes sense if e.g. std:: objects are passed
 * through between application and library, which is not the case for mkl-dnn
 * (since there is no any c++-rt dependent stuff, ideally...). */

/* SFINAE helper -- analogue to std::enable_if */
class VectorPrint {
public:
    template <typename Dtype>
    static void print_float(Dtype* target) {
        float* f = (float*)target;
        printf("size = %d\n", sizeof(Dtype));

        for (int i = 0; i < sizeof(Dtype) / sizeof(float); i++) {
            printf(" %f ,", f[i]);
        }

        printf("\n");
    }
};

class AlignedUtils {
public:
    template <typename Dtype>
    void aligned_last_dim(const Dtype* input, Dtype* output, int input_size, int ori_last_dim,
                          int aligned_dim) {
        for (int row = 0; row < input_size / ori_last_dim; row++) {
            for (int col = ori_last_dim; col < aligned_dim; col++) {
                output[row * aligned_dim + col] = static_cast<Dtype>(0);
            }
        }

        for (int i = 0; i < input_size; i++) {
            int row = i / ori_last_dim;
            int col = i % ori_last_dim;
            output[row * aligned_dim + col] = input[i];
        }
    }
    template <typename Dtype>
    void unaligned_last_dim(const Dtype* input, Dtype* output, int output_size, int ori_last_dim,
                            int aligned_dim) {
        for (int i = 0; i < output_size; i++) {
            int row = i / ori_last_dim;
            int col = i % ori_last_dim;
            output[i] = input[row * aligned_dim + col];
        }
    }

};

class SeqSortedseqTranseUtil {
public:
    SeqSortedseqTranseUtil(bool is_reverse = false, bool is_bi = false)
        : _is_reverse(is_reverse),
          _is_bi(is_bi) {};
    void print_vec(int* in, int size, const char* perfix) {
        for (int i = 0; i < size; i++) {
            printf("[%s] %d = %d\n", perfix, i, in[i]);
        }
    }
    template <typename Dtype>
    void seq_2_sorted_seq(const Dtype*  input, Dtype* output, int word_size) {
        //        _map_vec.resize(word_sum);
        int word_sum = _map_vec.size();
        //        std::cout << "word_sum = " << word_sum << std::endl;

        for (int ori_word_id = 0; ori_word_id < word_sum; ++ori_word_id) {
            //can param
            int word_start = ori_word_id * word_size;
            int maped_id = _map_vec[ori_word_id];
            int maped_start = maped_id * word_size;

            for (int word_vec_offset = 0; word_vec_offset < word_size; ++word_vec_offset) {
                //                std::cout<<maped_start + word_vec_offset<<" --> "<<word_start + word_vec_offset<<" , = "<<input[maped_start + word_vec_offset]<<std::endl;

                output[maped_start + word_vec_offset] = input[word_start + word_vec_offset];

            }
        }
    }
    template <typename Dtype>
    void hidden_2_sorted_hidden(const Dtype*  input, Dtype* output, int hidden_size) {
        //        _map_vec.resize(word_sum);
        int batch_size = _length_index.size();
        //        std::cout << "word_sum = " << word_sum << std::endl;

        for (int ori_word_id = 0; ori_word_id < batch_size; ++ori_word_id) {
            //can param
            int word_start = ori_word_id * hidden_size;
            int maped_id = _length_index[ori_word_id];
            int maped_start = maped_id * hidden_size;

            for (int word_vec_offset = 0; word_vec_offset < hidden_size; ++word_vec_offset) {
                //                std::cout<<maped_start + word_vec_offset<<" --> "<<word_start + word_vec_offset<<" , = "<<input[maped_start + word_vec_offset]<<std::endl;

                output[word_start + word_vec_offset] = input[maped_start + word_vec_offset];

            }
        }
    }
    template <typename Dtype>
    void sorted_seq_2_seq(const Dtype* input, Dtype* output, int hidden_size) {
        int word_sum = _map_vec.size();

        for (int ori_word_id = 0; ori_word_id < word_sum; ori_word_id++) {
            //can param
            int word_start = ori_word_id * hidden_size;
            int maped_id = _map_vec[ori_word_id];
            int maped_start = maped_id * hidden_size;

            for (int word_vec_offset = 0; word_vec_offset < hidden_size; word_vec_offset++) {
                //            std::cout<<ori_word_id+word_vec_offset<<" -> "<<maped_start+word_vec_offset<<std::endl;
                output[word_start + word_vec_offset] = input[maped_start + word_vec_offset];
            }
        }
    }
    template <typename Dtype>
    void sorted_seq_2_seq(const Dtype* input, Dtype* output, int hidden_size,
                          int alligned_hidden_size) {
        int word_sum = _map_vec.size();

        for (int ori_word_id = 0; ori_word_id < word_sum; ori_word_id++) {
            //can param
            int word_start = ori_word_id * hidden_size;
            int maped_id = _map_vec[ori_word_id];
            int maped_start = maped_id * alligned_hidden_size;

            for (int word_vec_offset = 0; word_vec_offset < hidden_size; word_vec_offset++) {
                //            std::cout<<ori_word_id+word_vec_offset<<" -> "<<maped_start+word_vec_offset<<std::endl;
                output[word_start + word_vec_offset] = input[maped_start + word_vec_offset];
            }
        }
    }
    /**
     * return whether need to transform
     * @param offset_vec
     * @param emit_offset_vec
     * @param emit_length
     * @return
     */
    bool get_sorted_map(std::vector<int>& offset_vec,
                        std::vector<int>& emit_offset_vec, int& emit_length, int skip_num = 0) {
        int batch_size = offset_vec.size() - 1;
        int word_sum = offset_vec[offset_vec.size() - 1];
        std::vector<int>length_vec(batch_size);
        _length_index.resize(batch_size);

        if (skip_num > 1) {
            CHECK_EQ(batch_size, 1) << "only support batch = 1 in skip_mode";
            CHECK_EQ(word_sum % skip_num, 0);
            int real_batch_size = skip_num;
            emit_length = word_sum / skip_num;
            emit_offset_vec.resize(emit_length + 1);
            emit_offset_vec[0] = 0;

            for (int i = 1; i <= emit_length; i++) {
                emit_offset_vec[i] = emit_offset_vec[i - 1] + skip_num;
            }

            return false;
        }

        if (batch_size == 1) {
            emit_length = offset_vec[1] - offset_vec[0];
            emit_offset_vec.resize(emit_length + 1);

            for (int i = 0; i <= emit_length; i++) {
                emit_offset_vec[i] = i;
            }

            return false;
        }

        int max_len = 0;

        for (int i = 0; i < offset_vec.size() - 1; ++i) {
            int len = offset_vec[i + 1] - offset_vec[i];
            max_len = max_len > len ? max_len : len;
            length_vec[i] = len;
            _length_index[i] = i;
        }

        emit_length = max_len;

        if (max_len == 1) {
            emit_offset_vec.push_back(0);
            emit_offset_vec.push_back(emit_length * batch_size);
            return false;
        }

        std::sort(_length_index.begin(), _length_index.end(), [&length_vec](int i1, int i2) {
            return length_vec[i1] > length_vec[i2];
        });

        emit_offset_vec.resize(max_len + 1);
        _map_vec.resize(word_sum);

        int target_word_id = 0;
        std::vector<int> length_vec_cnt = length_vec;
        int last_batch_size = batch_size;

        for (int word_id_in_seq = 0; word_id_in_seq < max_len; word_id_in_seq++) {
            emit_offset_vec[word_id_in_seq] = target_word_id;

            for (int batch_id = 0; batch_id < last_batch_size; batch_id++) {
                int old_batch_id = _length_index[batch_id];

                if (length_vec_cnt[old_batch_id] > 0) {
                    int inner_word_id_in_seq = word_id_in_seq;

                    if (_is_reverse) {
                        inner_word_id_in_seq = length_vec[old_batch_id] - 1 - word_id_in_seq;
                    }

                    int old_word_id = offset_vec[old_batch_id] + inner_word_id_in_seq;
                    _map_vec[old_word_id] = target_word_id;
                    //                    printf("map %d -> %d\n",old_word_id,target_word_id);
                    length_vec_cnt[old_batch_id]--;
                    target_word_id++;
                } else {
                    last_batch_size--;
                    break;
                }
            }
        }

        //        print_vec(_map_vec.data(),word_sum,"map");
        emit_offset_vec[max_len] = word_sum;
        return true;
    }


private:
    //    std::vector<int> _length_vec;
    std::vector<int> _length_index;
    std::vector<int> _map_vec;
    bool _is_reverse;
    bool _is_bi;

};

/* analogue std::conditional */
template <bool, typename, typename> struct conditional {};
template <typename T, typename F> struct conditional<true, T, F> {
    typedef T type;
};
template <typename T, typename F> struct conditional<false, T, F> {
    typedef F type;
};

template <bool, typename, bool, typename, typename> struct conditional3 {};
template <typename T, typename FT, typename FF>
struct conditional3<true, T, false, FT, FF> {
    typedef T type;
};
template <typename T, typename FT, typename FF>
struct conditional3<false, T, true, FT, FF> {
    typedef FT type;
};
template <typename T, typename FT, typename FF>
struct conditional3<false, T, false, FT, FF> {
    typedef FF type;
};

template <bool, typename U, U, U> struct conditional_v {};
template <typename U, U t, U f> struct conditional_v<true, U, t, f> {
    static constexpr U value = t;
};
template <typename U, U t, U f> struct conditional_v<false, U, t, f> {
    static constexpr U value = f;
};

template<typename T>
inline const T& min(const T& a, const T& b) {
    return a < b ? a : b;
}

template<typename T>
inline const T& max(const T& a, const T& b) {
    return a > b ? a : b;
}

template <typename T>
inline typename std::remove_reference<T>::type zero() {
    auto zero = typename std::remove_reference<T>::type();
    return zero;
}

template <typename T, typename P>
inline bool everyone_is(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
inline bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

template <typename T, typename P>
inline bool one_of(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename... Args>
inline bool any_null(Args... ptrs) {
    return one_of(nullptr, ptrs...);
}

inline bool implication(bool cause, bool effect) {
    return !cause || effect;
}

template<typename T>
inline void array_copy(T* dst, const T* src, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

template<typename T>
inline bool array_cmp(const T* a1, const T* a2, size_t size) {
    for (size_t i = 0; i < size; ++i) if (a1[i] != a2[i]) {
            return false;
        }

    return true;
}

template<typename T, typename U>
inline void array_set(T* arr, const U& val, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        arr[i] = static_cast<T>(val);
    }
}

namespace product_impl {

template<size_t> struct int2type {};

template <typename T>
constexpr int product_impl(const T* arr, int2type<0>) {
    return arr[0];
}

template <typename T, size_t num>
inline T product_impl(const T* arr, int2type<num>) {
    return arr[0] * product_impl(arr + 1, int2type < num - 1 > ());
}
}

template <size_t num, typename T>
inline T array_product(const T* arr) {
    return product_impl::product_impl(arr, product_impl::int2type < num - 1 > ());
}

template<typename T, typename R = T>
inline R array_product(const T* arr, size_t size) {
    R prod = 1;

    for (size_t i = 0; i < size; ++i) {
        prod *= arr[i];
    }

    return prod;
}

template <typename T, typename U>
inline typename std::remove_reference<T>::type div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline typename std::remove_reference<T>::type rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

template <typename T, typename U>
inline typename std::remove_reference<T>::type rnd_dn(const T a, const U b) {
    return (a / b) * b;
}

template <typename T, typename U, typename V>
inline U this_block_size(const T offset, const U max, const V block_size) {
    assert(offset < max);
    // TODO (Roma): can't use nstl::max() due to circular dependency... we
    // need to fix this
    const T block_boundary = offset + block_size;

    if (block_boundary > max) {
        return max - offset;
    } else {
        return block_size;
    }
}

template <typename Telem, size_t Tdims>
struct array_offset_calculator {
    template <typename... Targs>
    array_offset_calculator(Telem* base, Targs... Fargs) : _dims{ Fargs... } {
        _base_ptr = base;
    }

    template <typename... Targs>
    inline Telem& operator()(Targs... Fargs) {
        return *(_base_ptr + _offset(1, Fargs...));
    }

private:
    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t element) {
        return element;
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element) {
        return element + (_dims[dimension] * theta);
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element,
                          Targs... Fargs) {
        size_t t_prime = element + (_dims[dimension] * theta);
        return _offset(dimension + 1, t_prime, Fargs...);
    }

    Telem* _base_ptr;
    const int _dims[Tdims];
};

}//fin utils namespace

template<typename T> struct is_integral {
    static constexpr bool value = false;
};

template<> struct is_integral<int32_t> {
    static constexpr bool value = true;
};
template<> struct is_integral<int16_t> {
    static constexpr bool value = true;
};
template<> struct is_integral<int8_t> {
    static constexpr bool value = true;
};
template<> struct is_integral<uint8_t> {
    static constexpr bool value = true;
};

template <typename data_t, typename acc_t>
inline typename std::enable_if < !is_integral<data_t>::value,
       typename std::remove_reference<data_t>::type >::type
saturate(const acc_t& x) {
    return x;
}

template <typename data_t, typename acc_t>
inline typename std::enable_if<is_integral<data_t>::value,
       typename std::remove_reference<data_t>::type>::type
saturate(const acc_t& x) {
    acc_t v = x;

    if (v < (acc_t)std::numeric_limits<data_t>::lowest()) {
        v = (acc_t)std::numeric_limits<data_t>::lowest();
    }

    if (v > (acc_t)std::numeric_limits<data_t>::max()) {
        v = (acc_t)std::numeric_limits<data_t>::max();
    }

    return (typename std::remove_reference<data_t>::type)v;
}

template <typename out_t>
inline out_t round_and_saturate(float f, round_mode rmode) {
    switch (rmode) {
    case round_mode::nearest:
        f = nearbyintf(f);
        break;

    case round_mode::down:
        f = floorf(f);
        break;
    }

    return saturate<out_t>(f);
}

/* Quantization with beta == 0 */
template <typename in_t, typename out_t> struct qz_b0 {
    out_t operator()(in_t in, float alpha, round_mode rmode) {
        return round_and_saturate<out_t>(alpha * in, rmode);
    }
};

inline size_t datatype_size(DataType data_type) {
    switch (data_type) {
    case AK_FLOAT:
        return sizeof(float);

    case AK_INT32:
        return sizeof(int32_t);

    case AK_HALF:
        return sizeof(int16_t);

    case AK_INT8:
        return sizeof(int8_t);

    case AK_UINT8:
        return sizeof(uint8_t);

    case AK_INVALID:
    default:
        assert(!"unknown data_type");
    }

    return 0;
}

/** returns floor(log2(v)), aka the position of the leftmost non-0 bit */
inline int ilog2q(size_t v) {
    if (v == 0) {
        return -1;
    }

    int p = 0;
#   define CP(pw) do { if (v >= (1ull << pw)) { v >>= pw; p += pw; } } while(0)
    CP(32);
    CP(16);
    CP(8);
    CP(4);
    CP(2);
    CP(1);
#   undef CP
    return p;
}

struct scratchpad_t {
    virtual ~scratchpad_t() {}
    virtual char* get() const = 0;
};

template <typename T, typename U>
inline void balance2D(U nthr, U ithr, T ny, T& ny_start, T& ny_end,
                      T nx, T& nx_start, T& nx_end, T nx_divider) {
    const T grp_size = utils::div_up(nthr, nx_divider);
    const T grp_count = utils::div_up(nthr, grp_size);

    T grp = ithr / grp_size;
    T grp_ithr = ithr % grp_size;
    T grp_nthr = grp_size;
    T first_grps = nthr % grp_count;

    if (first_grps > 0 && grp >= first_grps) {
        ithr -= first_grps * grp_size;
        grp_nthr--;
        grp = ithr / grp_nthr + first_grps;
        grp_ithr = ithr % grp_nthr;
    }

    balance211(nx, grp_count, grp, nx_start, nx_end);
    balance211(ny, grp_nthr, grp_ithr, ny_start, ny_end);
}

template <typename T, typename U, typename V>
inline U this_block_size(const T offset, const U max, const V block_size) {
    assert(offset < max);
    // TODO (Roma): can't use nstl::max() due to circular dependency... we
    // need to fix this
    const T block_boundary = offset + block_size;

    if (block_boundary > max) {
        return max - offset;
    } else {
        return block_size;
    }
}

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_UTIL_H
