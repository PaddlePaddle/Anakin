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

template <typename HostType>
static void reorder_nhwc_nchw(const Tensor<HostType>& input,
                              Tensor<HostType>& output) {
    int n_value = input.num();
    int c_value = input.channel();
    int h_value = input.height();
    int w_value = input.width();

    if (input.get_layout() == Layout_NHWC && output.get_layout() == Layout_NCHW) {
        if (input.get_dtype() == AK_INT8 && output.get_dtype() == AK_FLOAT) {
            float* output_ptr = static_cast<float*>(output.mutable_data());
            CHECK(input.get_scale().size() >= 1);
            float scale = input.get_scale()[0];
            const int8_t* input_ptr = static_cast<const int8_t*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            int out_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            output_ptr[out_index] = input_ptr[in_index] * scale;
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_UINT8 && output.get_dtype() == AK_FLOAT) {
            LOG(INFO) << "print uint 8";
            CHECK(input.get_scale().size() >= 1);
            float scale = (input.get_scale()[0]) * (127.f / 255.f);
            LOG(INFO) << "scale = " << scale;
            double sum = 0.0;
            double max = 0.0;
            const uint8_t* input_ptr = static_cast<const uint8_t*>(input.data());
            float* output_ptr = static_cast<float*>(output.mutable_data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            int out_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            output_ptr[out_index] = (float)input_ptr[in_index] * scale;
                            sum += output_ptr[out_index];
                            max = output_ptr[out_index] > max ? output_ptr[out_index] : max;
                        }
                    }
                }
            }

            LOG(INFO) << "avg = " << (sum / input.valid_size()) << "," << max;
        } else if (input.get_dtype() == AK_UINT8 && output.get_dtype() == AK_UINT8) {
            LOG(INFO) << "reorder uint 8";
            uint8_t* output_ptr = static_cast<uint8_t*>(output.mutable_data());
            const uint8_t* input_ptr = static_cast<const uint8_t*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            int out_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            output_ptr[out_index] = input_ptr[in_index];
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_FLOAT) {
            const float* input_ptr = static_cast<const float*>(input.data());
            float* output_ptr = static_cast<float*>(output.mutable_data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            int out_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            output_ptr[out_index] = input_ptr[in_index];
                        }
                    }
                }
            }
        } else {
            LOG(FATAL) << "not support input type " << input.get_dtype();
        }
    } else if (input.get_layout() == Layout_NCHW && output.get_layout() == Layout_NHWC) {
        if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_FLOAT) {
            float* output_ptr = static_cast<float*>(output.mutable_data());
            const float* input_ptr = static_cast<const float*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            int out_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            output_ptr[out_index] = input_ptr[in_index];
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_UINT8 && output.get_dtype() == AK_UINT8) {
            uint8_t* output_ptr = static_cast<uint8_t*>(output.mutable_data());
            const uint8_t* input_ptr = static_cast<const uint8_t*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            int out_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            output_ptr[out_index] = input_ptr[in_index];
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_INT8 && output.get_dtype() == AK_INT8) {
            int8_t* output_ptr = static_cast<int8_t*>(output.mutable_data());
            const int8_t* input_ptr = static_cast<const int8_t*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            int out_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            output_ptr[out_index] = input_ptr[in_index];
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_INT8) {
            CHECK(output.get_scale().size() >= 1);
            float scale = 1.f / (output.get_scale()[0]);
            int8_t* output_ptr = static_cast<int8_t*>(output.mutable_data());
            const float* input_ptr = static_cast<const float*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            int out_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            output_ptr[out_index] = saturate<int8_t>(roundf(input_ptr[in_index] * scale));
                        }
                    }
                }
            }
        } else if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_UINT8) {
            CHECK(output.get_scale().size() >= 1);
            float scale = 1.f / (output.get_scale()[0]* (127.f / 255.f));
            uint8_t* output_ptr = static_cast<uint8_t*>(output.mutable_data());
            const float* input_ptr = static_cast<const float*>(input.data());

            for (int n = 0; n < n_value; ++n) {
                for (int c = 0; c < c_value; ++c) {
                    for (int h = 0; h < h_value; ++h) {
                        for (int w = 0; w < w_value; ++w) {
                            int in_index = n * c_value * h_value * w_value + c * h_value * w_value + h * w_value + w;
                            int out_index = n * h_value * w_value * c_value + h * w_value * c_value + w * c_value + c;
                            output_ptr[out_index] = saturate<uint8_t>(roundf(input_ptr[in_index] * scale));
                        }
                    }
                }
            }
        }else {
            LOG(FATAL) << "not support in/ou type " << input.get_dtype() << "," << output.get_dtype();
        }
    } else {
        LOG(FATAL) << "not support layout " << input.get_layout() << "," << output.get_layout();
    }

}

template <typename HostType>
static void reorder_nchwc_nchw(Tensor<HostType>& input,
                               Tensor<HostType>& output) {
    if (input.valid_shape() == output.valid_shape()) {
        output.copy_from(input);
        return;
    }

    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    LayoutType in_layout = input.get_layout();
    LayoutType out_layout = output.get_layout();
    bool is_nchwc_nchw = (in_layout == Layout_NCHW_C16R || in_layout == Layout_NCHW_C8R)
                         && (out_layout == Layout_NCHW);
    bool is_nchw_nchwc = (out_layout == Layout_NCHW_C16R || out_layout == Layout_NCHW_C8R)
                         && (in_layout == Layout_NCHW);
    CHECK(is_nchw_nchwc || is_nchwc_nchw) << "not support " << input.get_layout();

    if (is_nchwc_nchw) {
        Shape shape = output.valid_shape();
        int n_value = shape[0];
        int c_value = shape[1];
        int h_value = shape[2];
        int w_value = shape[3];
        Shape shape_input = input.valid_shape();
        int aligned_length = shape_input.get_layout_aligned_length();
        CHECK_GT(aligned_length, 0) << "input aligned should > 0";
        int c_round_divk = shape_input[1];

        c_round_divk = (shape_input.channel() + aligned_length - 1) / aligned_length;

        float* output_ptr = static_cast<float*>(output.mutable_data());
        const float* input_ptr = static_cast<const float*>(input.data());
        #pragma omp parallel for collapse(4) schedule(static)

        for (int n = 0; n < n_value; ++n) {
            for (int c = 0; c < c_value; ++c) {
                for (int h = 0; h < h_value; ++h) {
                    //#pragma ivdep
                    for (int w = 0; w < w_value; ++w) {
                        int round_c = c / aligned_length;
                        int remainder_c = c % aligned_length;
                        int input_idx = n * c_round_divk * h_value * w_value * aligned_length + round_c * h_value *
                                        w_value * aligned_length +
                                        h * w_value * aligned_length + w * aligned_length + remainder_c;
                        int output_idx = n * c_value * h_value * w_value + c * h_value * w_value  +
                                         h * w_value  + w ;

                        *(output_ptr + output_idx) = input_ptr[input_idx];
                    }
                }
            }
        }
    } else if (is_nchw_nchwc) {
        Shape shape = input.valid_shape();
        int n_value = shape[0], c_value = shape[1], h_value = shape[2], w_value = shape[3];

        int aligned_length = output.valid_shape().get_layout_aligned_length();
        CHECK_GT(aligned_length, 0) << "input aligned should > 0";

        int c_round_divk = (c_value + aligned_length - 1) / aligned_length;

        float* output_ptr = static_cast<float*>(output.mutable_data());
        const float* input_ptr = static_cast<const float*>(input.data());
        #pragma omp parallel for collapse(5) schedule(static)

        for (int n = 0; n < n_value; ++n) {
            for (int c_idx = 0; c_idx < c_round_divk; ++c_idx) {
                for (int h = 0; h < h_value; ++h) {
                    for (int w = 0; w < w_value; ++w) {
                        for (int c = 0; c < aligned_length; ++c) {
                            int input_idx = n * c_value * h_value * w_value + (c_idx * aligned_length + c) * h_value * w_value +
                                            h * w_value + w;
                            int output_idx = n * c_round_divk * h_value * w_value * aligned_length + c_idx * h_value * w_value *
                                             aligned_length +
                                             h * w_value * aligned_length + w * aligned_length + c;

                            *(output_ptr + output_idx) = ((c_idx * aligned_length + c) < c_value) ? *
                                                         (input_ptr + input_idx) : 0;
                        }
                    }
                }
            }
        }

    } else {
        LOG(FATAL) << "not support this shape";
    }


}

template <typename HostType>
static void reorder_nchwc8_nchw(Tensor<HostType>& input,
                                Tensor<HostType>& output) {

    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = output.valid_shape();
    int n_value = shape[0];
    int c_value = shape[1];
    int h_value = shape[2];
    int w_value = shape[3];
    Shape shape_input = input.valid_shape();
    int c_round_div8 = shape_input[1];

    if (input.get_layout() == Layout_NCHW_C8R) {
        c_round_div8 = (shape_input.channel() + 7) / 8;
    }

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
    #pragma omp parallel for collapse(4) schedule(static)

    for (int n = 0; n < n_value; ++n) {
        for (int c = 0; c < c_value; ++c) {
            for (int h = 0; h < h_value; ++h) {
                //#pragma ivdep
                for (int w = 0; w < w_value; ++w) {
                    int round_c = c / 8;
                    int remainder_c = c % 8;
                    int input_idx = n * c_round_div8 * h_value * w_value * 8 + round_c * h_value * w_value * 8 +
                                    h * w_value * 8 + w * 8 + remainder_c;
                    int output_idx = n * c_value * h_value * w_value + c * h_value * w_value  +
                                     h * w_value  + w ;

                    *(output_ptr + output_idx) = input_ptr[input_idx];
                }
            }
        }
    }
}

template <typename HOST_TYPE>
inline void calibrate_int8c4_to_fp32_host(Tensor<HOST_TYPE>& host_tensor,
        const Tensor <HOST_TYPE>& int8_tensor) {

    CHECK_EQ(host_tensor.get_dtype(), AK_FLOAT);
    CHECK_EQ(host_tensor.get_layout(), Layout_NCHW);
    CHECK_EQ(int8_tensor.get_dtype(), AK_INT8);
    CHECK_EQ(int8_tensor.get_layout(), Layout_NCHW_C4);
    CHECK_EQ(host_tensor.valid_size(), int8_tensor.valid_size());
    CHECK_GE(int8_tensor.get_scale().size(), 1);

    Shape out_stride = host_tensor.get_stride();
    Shape in_shape = int8_tensor.valid_shape();
    Shape out_shape = host_tensor.valid_shape();
    int valid_width = in_shape.width();
    int valid_height = in_shape.height();
    int valid_channel_4 = in_shape.channel() / 4;
    int valid_num = in_shape.num();
    int in_n_stride = in_shape[1] * in_shape[2] * in_shape[3] / 4;
    int in_c_stride = in_shape[2] * in_shape[3];
    int in_h_stride = in_shape[3];
    int in_w_stride = 1;

    int count = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] / 4;
    const char* in_data = (const char*)int8_tensor.data();
    float* out_data = (float*)host_tensor.mutable_data();
    float scale = int8_tensor.get_scale()[0];

    for (int gid = 0; gid < count; ++ gid) {
        float load0, load1, load2, load3;

        int read_w = (gid) % valid_width;
        int read_h = (gid / (in_h_stride)) % valid_height;
        int read_c = (gid / (in_c_stride)) % valid_channel_4;
        int read_n = (gid / (in_n_stride)) % valid_num;

        int in_offset = read_n * in_n_stride
                        + read_c * in_c_stride
                        + read_h * in_h_stride
                        + read_w;

        int out_offset = read_n * out_stride[0]
                         + read_c * (out_stride[1] << 2)
                         + read_h * out_stride[2]
                         + read_w * out_stride[3];

        if (gid < count) {

            char readin0 = in_data[4 * in_offset + 0];
            char readin1 = in_data[4 * in_offset + 1];
            char readin2 = in_data[4 * in_offset + 2];
            char readin3 = in_data[4 * in_offset + 3];

            load0 = static_cast<float>(readin0);
            load1 = static_cast<float>(readin1);
            load2 = static_cast<float>(readin2);
            load3 = static_cast<float>(readin3);

            out_data[out_offset] = load0 * scale;
            out_offset += out_stride[1];
            out_data[out_offset] = load1 * scale;
            out_offset += out_stride[1];
            out_data[out_offset] = load2 * scale;
            out_offset += out_stride[1];
            out_data[out_offset] = load3 * scale;
        }
    }
}

}
}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_UTIL_H
