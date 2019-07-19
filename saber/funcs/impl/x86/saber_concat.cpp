#include "saber/funcs/impl/x86/saber_concat.h"

namespace anakin {

namespace saber {

template <typename dtype>
void concat_kernel(const int len, const dtype* src, dtype* dst) {
    if (dst != src) {
        memcpy(dst, src, sizeof(dtype) * len);
    }
}
template <>
SaberStatus SaberConcat<X86, AK_FLOAT>::create(const std::vector<Tensor<X86>*>& inputs,
                   std::vector<Tensor<X86>*>& outputs,
                   ConcatParam<X86> &param, Context<X86> &ctx){

    _num_concats = inputs[0]->count_valid(0, param.axis);
    _concat_input_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
    return SaberSuccess;
}

template <>
SaberStatus SaberConcat<X86, AK_FLOAT>::dispatch(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs, ConcatParam<X86>& param) {

    int input_size = inputs.size();
    //! get output data, valid shape and stride shape
    int offset_concat_axis = 0;
    Shape out_shape = outputs[0]->valid_shape();
    const int out_concat_axis = out_shape[param.axis];

    if (inputs[0]->get_layout() == Layout_NCHW_C8R) {
        for (int i = 1; i < input_size; i++) {
            CHECK_EQ(inputs[i]->get_layout(), Layout_NCHW_C8R) << "concat layout should euqal";
        }

        CHECK_EQ(outputs[0]->get_layout(), Layout_NCHW_C8R) << "concat output layout should euqal";

        if (inputs.size() == 1) {
            outputs[0]->copy_from(*inputs[0]);
            return SaberSuccess;
        }

        OpDataType* dout = (OpDataType*)outputs[0]->mutable_data();

        for (int i = 0; i < input_size; ++i) {
            Shape sh_in = inputs[i]->valid_shape();
            const OpDataType* din = (const OpDataType*)inputs[i]->data();
            const int in_concat_axis = sh_in[param.axis];

            for (int n = 0; n < _num_concats; ++n) {
                concat_kernel<OpDataType>(in_concat_axis * _concat_input_size,
                                          din + n * in_concat_axis * _concat_input_size,
                                          dout + (n * out_concat_axis + offset_concat_axis)
                                          * _concat_input_size);
            }

            offset_concat_axis += in_concat_axis;
        }

        outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
        return SaberSuccess;
    }

    if (inputs.size() == 1) {
        outputs[0]->copy_from(*inputs[0]);
        return SaberSuccess;
    }

    OpDataType* dout = (OpDataType*)outputs[0]->mutable_data();

    for (int i = 0; i < input_size; ++i) {
        Shape sh_in = inputs[i]->valid_shape();
        const OpDataType* din = (const OpDataType*)inputs[i]->data();
        const int in_concat_axis = sh_in[param.axis];

        for (int n = 0; n < _num_concats; ++n) {
            concat_kernel<OpDataType>(in_concat_axis * _concat_input_size,
                                      din + n * in_concat_axis * _concat_input_size,
                                      dout + (n * out_concat_axis + offset_concat_axis)
                                      * _concat_input_size);
        }

        offset_concat_axis += in_concat_axis;
    }

    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;
}

template <>
SaberStatus SaberConcat<X86, AK_INT8>::create(const std::vector<Tensor<X86>*>& inputs,
                                              std::vector<Tensor<X86>*>& outputs,
                                              ConcatParam<X86> &param,
                                              Context<X86> &ctx) {
    SaberStatus status = SaberSuccess;
    jit::jit_concat_conf_t jpp_;
    float *src_scale;
    Tensor<X86>* input = inputs[0];
    Tensor<X86>* output = outputs[0];

    if (input == nullptr || output == nullptr) {
        return SaberInvalidValue;
    }

    if (inputs[0]->get_dtype() == AK_UINT8) {
        Shape src_shape(input->shape());
        Shape dst_shape(output->shape());
        memset(&jpp_, 0, sizeof(jit::jit_concat_conf_t));

        jpp_.bs = src_shape[0];
        jpp_.oc = dst_shape[3];
        jpp_.h = src_shape[1];
        jpp_.w = src_shape[2];

        jpp_.n_inputs = inputs.size();
        jpp_.src_dt = inputs[0]->get_dtype();
        jpp_.dst_dt = outputs[0]->get_dtype();;
        jpp_.with_relu = false;

        const int num_srcs = jpp_.n_inputs;
        srcs_data_ = (const unsigned char**)zmalloc(num_srcs * sizeof(unsigned char *), 4096);
        nb_ic_ = (unsigned int* )zmalloc(num_srcs * sizeof(unsigned int), 4096);
        scale_ = (float* )zmalloc(num_srcs * sizeof(float), 4096);
        block_ = (unsigned int* )zmalloc(num_srcs * sizeof(unsigned int), 4096);
        tail_ = (unsigned long* )zmalloc(num_srcs * sizeof(unsigned long), 4096);
        ic_ = (unsigned int* )zmalloc(num_srcs * sizeof(unsigned int), 4096);

        if ((srcs_data_ == nullptr) ||
            (nb_ic_ == nullptr) ||
            (scale_ == nullptr) ||
            (block_ == nullptr) ||
            (tail_ == nullptr) ||
            (ic_ == nullptr)) {
            if (srcs_data_) {
                delete srcs_data_;
                srcs_data_ = nullptr;
            }
            if (nb_ic_) {
                delete nb_ic_;
                nb_ic_ = nullptr;
            }
            if (scale_) {
                delete scale_;
                scale_ = nullptr;
            }
            if (block_) {
                delete block_;
                block_ = nullptr;
            }
            if (tail_) {
                delete tail_;
                tail_ = nullptr;
            }
            if (ic_) {
                delete ic_;
                ic_ = nullptr;
            }
            return SaberOutOfMem;
        }

        if (inputs[0]->get_scale().data() != nullptr) {
            jpp_.scale_max = inputs[0]->get_scale()[0];
        } else {
            jpp_.scale_max = 1.0f;
        }

        for (int i = 0; i < num_srcs; i++) {
            src_scale = (float*)inputs[i]->get_scale().data();
            if (src_scale != nullptr) {
                if (jpp_.scale_max < inputs[i]->get_scale()[0]) {
                    jpp_.scale_max = inputs[i]->get_scale()[0];
                }
            } else {
                jpp_.scale_max = 1.0f;
            }
        }

        for (int i = 0; i < num_srcs; ++i) {
            Shape src_shape_temp(inputs[i]->shape());
            ic_[i] = src_shape_temp[3];

            if ((src_scale != nullptr) &&
                (fabs(jpp_.scale_max - inputs[i]->get_scale()[0]) > FLT_MIN)) {
                block_[i] = 512 / sizeof(float) / 8;
                nb_ic_[i] = src_shape_temp[3] / block_[i];
                tail_[i] = (1ULL << (src_shape_temp[3] % block_[i])) -1;
            } else {
                block_[i] = 512 / sizeof(unsigned char) / 8;
                nb_ic_[i] = src_shape_temp[3] / block_[i];
                tail_[i] = (1ULL << (src_shape_temp[3] % (block_[i] * sizeof(unsigned char)))) -1;
            }

            srcs_data_[i] = (unsigned char *)(inputs[i]->data());
            src_scale = inputs[i]->get_scale().data();
            if (src_scale != nullptr) {
                scale_[i] = jpp_.scale_max / inputs[i]->get_scale()[0];
            } else {
                scale_[i] = jpp_.scale_max;
            }
        }

        jpp_.nb_ic = nb_ic_;
        jpp_.block = block_;
        jpp_.tail = tail_;
        jpp_.ic = ic_;

        dst_data_ = (unsigned char *)outputs[0]->data();
        jpp_.scales = scale_;

        const int nthreads = anakin_get_max_threads();
        src_with_offset_ = (const unsigned char **)zmalloc(nthreads * num_srcs * sizeof(DataType *), 4096);

        status = jit::jit_avx512_core_8bit_concat_kernel::init_conf(jpp_);
        if (status == SaberSuccess) {
            if (kernel != nullptr) {
                delete kernel;
                kernel = nullptr;
            }
            kernel = new jit::jit_avx512_core_8bit_concat_kernel(jpp_);
        } else {
            return SaberUnImplError;
        }
    } else {
        _num_concats = inputs[0]->count_valid(0, param.axis);
        _concat_input_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberConcat<X86, AK_INT8>::dispatch(const std::vector<Tensor<X86>*>& inputs,
                                                std::vector<Tensor<X86>*>& outputs,
                                                ConcatParam<X86> &param) {
    int input_size = inputs.size();
    int offset_concat_axis = 0;
    Shape out_shape = outputs[0]->valid_shape();
    const int out_concat_axis = out_shape[param.axis];

    if (inputs.size() == 1) {
        outputs[0]->copy_from(*inputs[0]);
        return SaberSuccess;
    }

    if (inputs[0]->get_dtype() == AK_UINT8) {
        const auto &jpp = kernel->jpp;
        const int work_amount = jpp.bs * jpp.h * jpp.w;
        const int max = anakin_get_max_threads();

        if (work_amount < max) {
#pragma omp parallel for num_threads(work_amount)
            for (int iwork = 0; iwork < work_amount; iwork++) {
                auto srcs = src_with_offset_ + iwork * jpp.n_inputs;
                for (int i = 0; i < jpp.n_inputs; i++) {
                    srcs[i] = (const unsigned char*)srcs_data_[i] + iwork * ic_[i];
                }
                jit::jit_concat_call_t p;;
                memset(&p, 0, sizeof(jit::jit_concat_call_t));

                p.src = reinterpret_cast<const unsigned char**>(srcs);
                p.dst = reinterpret_cast<unsigned char*>(dst_data_ + iwork * jpp.oc);
                kernel->ker_(&p);
            }
        } else {
#pragma omp parallel
            {
                int ithr = anakin_get_thread_num(), nthr = anakin_get_num_threads();
                int start{0}, end{0};
                balance211(work_amount, nthr, ithr, start, end);

                int n{0}, h{0}, w{0};
                nd_iterator_init(start, n, jpp.bs, h, jpp.h, w, jpp.w);
                auto srcs = src_with_offset_ + ithr * jpp.n_inputs;

                jit::jit_concat_call_t p;
                memset(&p, 0, sizeof(jit::jit_concat_call_t));

                for (int iwork = start; iwork < end; ++iwork) {
                    int nhw = n * (jpp.h * jpp.w) + h * (jpp.w) + w;
                    for (int i = 0; i < jpp.n_inputs; ++i) {
                        srcs[i] = srcs_data_[i] + (nhw * ic_[i]);
                    }

                    p.src  = reinterpret_cast<const unsigned char **>(srcs);
                    p.dst  = reinterpret_cast<unsigned char *>(dst_data_ + iwork * jpp.oc);

                    // one kernel move one dst oc from all srcs
                    kernel->ker_(&p);
                    nd_iterator_step(n, jpp.bs, h, jpp.h, w, jpp.w);
                }
            }
        }
    } else {
        OpDataType* dout = (OpDataType*)outputs[0]->mutable_data();
        for (int i = 0; i < input_size; ++i) {
            Shape sh_in = inputs[i]->valid_shape();
            const OpDataType* din = (OpDataType*)inputs[i]->data();
            const int in_concat_axis = sh_in[param.axis];
            for (int n = 0; n < _num_concats; ++n) {
                concat_kernel<OpDataType>(in_concat_axis * _concat_input_size,
                                          din + n * in_concat_axis * _concat_input_size,
                                          dout + (n * out_concat_axis + offset_concat_axis)
                                                 * _concat_input_size);
            }
            offset_concat_axis += in_concat_axis;
        }
        outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    }
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberConcat<X86, OpDtype>::init_conf(jit::jit_concat_conf_t &jpp,
                                                 const std::vector<Tensor<X86>*> &inputs,
                                                 std::vector<Tensor<X86>*> &outputs,
                                                 ConcatParam<X86> &param){
    return SaberSuccess;
};

template <DataType OpDtype>
SaberStatus SaberConcat<X86, OpDtype>::check_conf(const jit::jit_concat_conf_t &jpp,
                                                  const std::vector<Tensor<X86>*> &inputs,
                                                  std::vector<Tensor<X86>*> &outputs,
                                                  ConcatParam<X86> &param){
    return SaberSuccess;
};
template <>
SaberStatus SaberConcat<X86, AK_INT8>::init_conf(jit::jit_concat_conf_t &jpp,
                                                 const std::vector<Tensor<X86>*> &inputs,
                                                 std::vector<Tensor<X86>*> &outputs,
                                                 ConcatParam<X86> &param) {
    using namespace utils;

    Tensor<X86>* input = inputs[0];
    Tensor<X86>* output = outputs[0];

    bool ok = true
              && jit::mayiuse(jit::avx512_common)
              && (inputs[0]->get_layout() ==  Layout_NHWC)
              && (outputs[0]->get_layout() == Layout_NHWC);
    if (!ok) {
                LOG(ERROR) << "unimplement error!"
                           << " mayiuse(avx512_common) :" << jit::mayiuse(jit::avx512_common)
                           << " std::is_same<inputs[0]->get_layout(), Layout_NHWC>::value :"
                           << (inputs[0]->get_layout() ==  Layout_NHWC)
                           << " std::is_same<outputs[0]->get_layout(), Layout_NHWC>::value :"
                           << (outputs[0]->get_layout() == Layout_NHWC);
        return SaberUnImplError;
    }

    if (jit::jit_avx512_core_8bit_concat_kernel::init_conf(jpp)) {
        return SaberSuccess;
    } else {
                LOG(ERROR) << "kernel init failed!";
        return SaberUnImplError;
    }
}

template <>
SaberStatus SaberConcat<X86, AK_INT8>::check_conf(const jit::jit_concat_conf_t &jpp,
                                                  const std::vector<Tensor<X86>*> &inputs,
                                                  std::vector<Tensor<X86>*> &outputs,
                                                  ConcatParam<X86> &param) {
    using namespace utils;

    Tensor<X86>* input = inputs[0];
    Tensor<X86>* output = outputs[0];

    bool ok = true
              && jit::mayiuse(jit::avx512_common)
              && (inputs[0]->get_layout() ==  Layout_NHWC)
              && (outputs[0]->get_layout() == Layout_NHWC);
    if (!ok) {
                LOG(ERROR) << "unimplement error!"
                           << " mayiuse(avx512_common) :" << jit::mayiuse(jit::avx512_common)
                           << " std::is_same<inputs[0]->get_layout(), Layout_NHWC>::value :"
                           << (inputs[0]->get_layout() ==  Layout_NHWC)
                           << " std::is_same<outputs[0]->get_layout(), Layout_NHWC>::value :"
                           << (outputs[0]->get_layout() == Layout_NHWC);
        return SaberUnImplError;
    }

    return SaberSuccess;
}


template class SaberConcat<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, X86, AK_HALF);
} //namespace anakin

} //namespace anakin
