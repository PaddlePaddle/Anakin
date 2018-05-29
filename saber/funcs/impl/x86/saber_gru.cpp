

#include "saber/funcs/impl/x86/saber_gru.h"
#include "saber/core/tensor_op.h"
#include "cuda_fp16.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "mkl_cblas.h"
#include <malloc.h>
namespace anakin {

namespace saber {

static void write_tensorfile(std::vector<float>& f, const char* locate) {
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        std::cout << "file open field :" << locate << std::endl;
    } else {
        for (int i = 0; i < f.size(); ++i) {
            fprintf(fp, "[%d] %f \n", i, (f[i]));
        }

        fclose(fp);
    }

    std::cout << "!!! write success: " << locate << " , size = " << f.size() << std::endl;
}

static void write_tensorfile(const float* f, const char* locate, int size) {
    FILE* fp = fopen(locate, "w+");

    if (fp == 0) {
        std::cout << "file open field :" << locate << std::endl;
    } else {
        for (int i = 0; i < size; ++i) {
            fprintf(fp, "[%d] %f \n", i, (f[i]));
        }

        fclose(fp);
    }

    std::cout << "!!! write success: " << locate << " , size = " << size << std::endl;
}

template <typename Dtype>
void tensor_cmp_host(const Dtype* src1, const Dtype* src2, int size) {


    double max_ratio;
    double max_diff;
    const double eps = 1e-6f;
    max_diff = fabs(src1[0] - src2[0]);
    max_ratio = 2.0 * max_diff / (src1[0] + src2[0] + eps);

    for (int i = 1; i < size; ++i) {
        double diff = fabs(src1[i] - src2[i]);

        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (src1[i] + src2[i] + eps);
        }
    }

    if (abs(max_ratio) > 0.001) {
        std::cout << "test failed " << abs(max_ratio) << "  <=  " << max_diff << std::endl;
    } else {
        std::cout << "passed! " << abs(max_ratio) << "  <=  " << max_diff << std::endl;
    }
}

static void* aligned_malloc(size_t align, size_t size) {
    size *= 4;
    void* result = NULL;
#ifdef _MSC_VER
    result = _aligned_malloc(size, align);
#elif __APPLE__

    if (posix_memalign(&result, align, size)) {
        result = NULL;
    }

#elif __linux__
    result = memalign(align, size);
#endif
    return result;
}


static void gemm(const bool TransA, const bool TransB, int m, int n, int k, const float alpha,
                 const float* a, const float* b, const float beta, float* c) {
    //    cout << "(" << m << "," << n << "," << k << ")" << endl;
    int lda = (!TransA/* == CblasNoTrans*/) ? k : m;
    int ldb = (!TransB/* == CblasNoTrans*/) ? n : k;
    CBLAS_TRANSPOSE cuTransA =
        (!TransA/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cuTransB =
        (!TransB/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);
};

template <typename Dtype>
Dtype Sigmoid(const Dtype a) {
    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-a));
}

//template<>
//void SaberGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::
//lod_no_batch_gru(const OpDataType* weight_w, const OpDataType* weight_h,const OpDataType* b, const OutDataType* h_init, OutDataType* h_out,
//                 const InDataType* x,OutDataType *temp_wx,OutDataType *temp_wh,OutDataType *temp_whr,
//                 int hidden_size, int word_size, std::vector<int>& offset_vec, bool is_reverse){
//    std::vector<int> length_vec(offset_vec.size() - 1);
//    int batch_size = offset_vec.size() - 1;
//    int seqsum = 0;
//    int max_seq_len = 0;
//
//    for (int i = 0; i < offset_vec.size() - 1; ++i) {
//        int len = offset_vec[i + 1] - offset_vec[i];
//        max_seq_len = max_seq_len > len ? max_seq_len : len;
//        length_vec[i] = len;
//        seqsum += len;
//    }
//
//    std::vector<int> seqid2batchid(seqsum);
//
//    for (int batchid = 0; batchid < batch_size; ++batchid) {
//        for (int i = offset_vec[batchid]; i < offset_vec[batchid + 1]; ++i) {
//            seqid2batchid[i] = batchid;
//        }
//    }
//
//    //wx
//    gemm(false, false, seqsum, 3 * hidden_size, word_size, 1.f, x, weight_w, 0.f, temp_wx);
//
//    int o_offset = 0;
//    int r_offset = 1;
//    int z_offset = 2;
//    const OpDataType *b_r = b + r_offset * hidden_size;
//    const OpDataType *b_z = b + z_offset * hidden_size;
//    const OpDataType *b_o = b + o_offset * hidden_size;
//
//    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
//        int batch_offset = offset_vec[batch_id];
//        int batch_length = length_vec[batch_id];
//
//        for (int seq_id_in_batch = 0; seq_id_in_batch < length_vec[batch_id]; ++seq_id_in_batch) {
//            int seqid = batch_offset + seq_id_in_batch;
//            int last_seq_id = seqid - 1;
//
//            if (is_reverse) {
//                seqid = batch_offset + batch_length - 1 - seq_id_in_batch;
//                last_seq_id = seqid + 1;
//            }
//
//            const OutDataType *hin;
//            OutDataType *hout = seqid * hidden_size + h_out;
//
//            if (seq_id_in_batch == 0) {
//                hin = h_init + batch_id * hidden_size;
//
//            } else {
//                hin = h_out + last_seq_id * hidden_size;
//            }
//
//            gemm(false, false, 1, 2 * hidden_size, hidden_size, 1.0, hin, weight_h + hidden_size * hidden_size,
//                 0.f, temp_wh);
//
//            OutDataType r;
//            OutDataType z;
//            OutDataType _h;
//            OutDataType *w_x_r = temp_wx+ r_offset * hidden_size
//                           + seqid * hidden_size * 3;
//            OutDataType *w_x_z = temp_wx+ z_offset * hidden_size
//                           + seqid * hidden_size * 3;
//            OutDataType *w_x_o = temp_wx + o_offset * hidden_size
//                           + seqid * hidden_size * 3;
//
//            OutDataType *w_h_r = temp_wh + 0 * hidden_size;
//            OutDataType *w_h_z = temp_wh + 1 * hidden_size;
//            OpDataType *w_o = weight_h;
//
//            for (int frame_id = 0; frame_id < hidden_size; ++frame_id) {
//                r = w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]; //h_out=gate_r
//                r = Sigmoid(r);
//                hout[frame_id] = r * hin[frame_id];
//            }
//
//            gemm(false, false, 1, hidden_size, hidden_size, 1.0, hout, w_o, 0.f, temp_whr);
//
//            for (int frame_id = 0; frame_id < hidden_size; ++frame_id) {
//                z = Sigmoid(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
//                _h = w_x_o[frame_id] + temp_whr[frame_id] + b_o[frame_id];
//                _h = tanh(_h);
//                hout[frame_id] = (1 - z) * hin[frame_id] + z * _h;
//            }
//        }
//    }
//}

template<>
SaberStatus SaberGru<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW_C16, NCHW_C16, NCHW_C16>::dispatch(\
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        GruParam<OpTensor>& param) {
    CHECK_NE(param._formula, GRU_CUDNN) << "X86 gru not support cudnn formula now";
    const OpDataType* weight_h = _weights_h2h.data();
    const OpDataType* weight_w = _weights_i2h.data();
    const OpDataType* bias = _weights_bias.data();
    std::vector<std::vector<int> >lod=inputs[0]->get_seq_offset();
    std::vector<int> offset_vec = lod[lod.size()-1];
    bool is_hw2seq = offset_vec.size() > 2;
    int word_sum = is_hw2seq ? offset_vec[offset_vec.size() - 1] : inputs[0]->channel();
    const OutDataType* h_init = nullptr;

    if (inputs.size() > 1) {
        h_init = inputs[1]->data();
    } else if (param.init_hidden() != nullptr) {
        h_init = param.init_hidden()->data();
    }

    const InDataType* x = inputs[0]->data();
    OutDataType* out = outputs[0]->mutable_data();

    //    lod_no_batch_gru(weight_w,weight_h,bias,h_init,out,x
    //            ,_temp_wx.mutable_data(),_temp_wh.mutable_data(),_temp_whr.mutable_data(),
    //                     _hidden_size,_word_size,offset_vec,param._is_reverse);
    {
        bool is_reverse = param._is_reverse;
        OutDataType* temp_wh = _temp_wh.mutable_data();
        OutDataType* temp_wx = _temp_wx.mutable_data();
        OutDataType* temp_whr = _temp_whr.mutable_data();

        std::vector<int> length_vec(offset_vec.size() - 1);
        int batch_size = offset_vec.size() - 1;
        int seqsum = 0;
        int max_seq_len = 0;

        for (int i = 0; i < offset_vec.size() - 1; ++i) {
            int len = offset_vec[i + 1] - offset_vec[i];
            max_seq_len = max_seq_len > len ? max_seq_len : len;
            length_vec[i] = len;
            seqsum += len;
        }

        std::vector<int> seqid2batchid(seqsum);

        for (int batchid = 0; batchid < batch_size; ++batchid) {
            for (int i = offset_vec[batchid]; i < offset_vec[batchid + 1]; ++i) {
                seqid2batchid[i] = batchid;
            }
        }

        //wx
        gemm(false, false, seqsum, 3 * _hidden_size, _word_size, 1.f, x, weight_w, 0.f, temp_wx);

        int o_offset = 0;
        int r_offset = 1;
        int z_offset = 2;
        const OpDataType* b_r = bias + r_offset * _hidden_size;
        const OpDataType* b_z = bias + z_offset * _hidden_size;
        const OpDataType* b_o = bias + o_offset * _hidden_size;


        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            int batch_offset = offset_vec[batch_id];
            int batch_length = length_vec[batch_id];

            for (int seq_id_in_batch = 0; seq_id_in_batch < length_vec[batch_id]; ++seq_id_in_batch) {
                int seqid = batch_offset + seq_id_in_batch;
                int last_seq_id = seqid - 1;

                if (is_reverse) {
                    seqid = batch_offset + batch_length - 1 - seq_id_in_batch;
                    last_seq_id = seqid + 1;
                }

                const OutDataType* hin;
                OutDataType* hout = seqid * _hidden_size + out;

                if (seq_id_in_batch == 0) {
                    hin = h_init + batch_id * _hidden_size;

                } else {
                    hin = out + last_seq_id * _hidden_size;
                }

                gemm(false, false, 1, 2 * _hidden_size, _hidden_size, 1.0, hin,
                     weight_h + _hidden_size * _hidden_size,
                     0.f, temp_wh);

                OutDataType r;
                OutDataType z;
                OutDataType _h;
                OutDataType* w_x_r = temp_wx + r_offset * _hidden_size
                                     + seqid * _hidden_size * 3;
                OutDataType* w_x_z = temp_wx + z_offset * _hidden_size
                                     + seqid * _hidden_size * 3;
                OutDataType* w_x_o = temp_wx + o_offset * _hidden_size
                                     + seqid * _hidden_size * 3;

                OutDataType* w_h_r = temp_wh + 0 * _hidden_size;
                OutDataType* w_h_z = temp_wh + 1 * _hidden_size;
                OpDataType* w_o = weight_h;

                for (int frame_id = 0; frame_id < _hidden_size; ++frame_id) {
                    r = w_x_r[frame_id] + w_h_r[frame_id] + b_r[frame_id]; //h_out=gate_r
                    r = Sigmoid(r);
                    hout[frame_id] = r * hin[frame_id];
                }

                gemm(false, false, 1, _hidden_size, _hidden_size, 1.0, hout, w_o, 0.f, temp_whr);

                for (int frame_id = 0; frame_id < _hidden_size; ++frame_id) {
                    z = Sigmoid(w_x_z[frame_id] + w_h_z[frame_id] + b_z[frame_id]);
                    _h = w_x_o[frame_id] + temp_whr[frame_id] + b_o[frame_id];
                    _h = tanh(_h);
                    hout[frame_id] = (1 - z) * hin[frame_id] + z * _h;
                }
            }
        }
    }

};
        template class SaberGru<X86,AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW_C16, NCHW_C16, NCHW_C16>;
}
}