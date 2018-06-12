#ifndef ANAKIN_SABER_FUNC_IMPL_X86_MATH_SEQUENCE_BATCH_H
#define ANAKIN_SABER_FUNC_IMPL_X86_MATH_SEQUENCE_BATCH_H

#include <algorithm>
#include <vector>
#include "saber/core/tensor.h"

namespace anakin {
namespace saber {
namespace math {

template <DataType Dtype, typename LayOutType>
class CopyMatrixRowsFunctor {
public:
    typedef Tensor<X86, Dtype, LayOutType> ioTensor;
    typedef typename ioTensor::Dtype dtype;

    // If is_src_index is true,
    // copy the indexed rows of input src to the output dst.
    // If is_src_index is false,
    // copy the input src to the indexed rows of output dst.
    // The indexed rows are based on the input index.
    void operator()(ioTensor* src,
                  std::vector<int> index_lod, ioTensor* dst,
                  bool is_src_index);
};

template <DataType Dtype, typename LayOutType>
class LoDTensor2BatchFunctor {
    // Calculate the length of each sequence and
    // sort sequence index by the length.
    // example:  sequences = {s0, s1, s2}
    //           s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
    //           seq_info[3] = {(4, 5, 1), (0, 4, 0), (9, 3, 2)}
    //
    struct SeqInfo {
        SeqInfo(int start, int length, int seq_idx)
            : start(start), length(length), seq_idx(seq_idx) {}
        int start;
        int length;
        int seq_idx;
    };

public:
    typedef Tensor<X86, Dtype, LayOutType> ioTensor;
    void operator()(ioTensor* seq,
                  ioTensor* batch, std::vector<std::vector<int>>& seq_to_batch_meta, bool is_cal_batch_lod,
                  bool is_reverse = false) const {
        if (!is_cal_batch_lod) {
            if (seq_to_batch_meta.size() < 2) {
                LOG(ERROR) << "The size of seq_to_batch_meta should inlcude at least 2-level sequence information.";
                exit(-1);
            }
            if (seq_to_batch_meta[1].size() != static_cast<int>(seq->num())) {
                LOG(ERROR) << "The seq_to_batch information should be consistent with the dims.";
                exit(-1);
            }
            CopyMatrixRowsFunctor<Dtype, LayOutType> to_batch;
            to_batch(seq, seq_to_batch_meta[1], batch, true);
            return;
        }

        if (seq_to_batch_meta.size() != 1) {
            LOG(ERROR) << "Only support one level sequence now.";
            exit(-1);
        }

        auto seq_meta = seq_to_batch_meta[0];

        std::vector<SeqInfo> seq_info;
        for (int seq_id = 0; seq_id < seq_meta.size() - 1; ++seq_id) {
            int length = seq_meta[seq_id + 1] - seq_meta[seq_id];
            seq_info.emplace_back(seq_meta[seq_id], length, seq_id);
            //LOG(INFO) << "seq_meta[seq_id]:" << seq_meta[seq_id] << " length:" << length << " seq_id:" <<seq_id;
        }

        std::sort(seq_info.begin(), seq_info.end(),
              [](SeqInfo a, SeqInfo b) { return a.length > b.length; });

        // Calculate the start position of each batch.
        // example:  sequences = {s0, s1, s2}
        //           s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
        //           num_batch = 5,
        //           batchIndex = {b0, b1, b2, b3, b4}
        //           b0: 1 0 2, b1: 1 0 2, b2: 1 0 2, b3: 1 0, b4: 1
        //           batch_start_positions[6] = {0, 3, 6, 9, 11, 12}
        //              batch_start_positions[0] = len(b0)
        //              batch_start_positions[1] = len(b0) + len(b1)
        //              batch_start_positions[2] = len(b0) + len(b1) + len(b2)
        //              ...
        //           seq2batch_idx[12] = {4, 0, 9,
        //                                5, 1, 10,
        //                                6, 2, 11,
        //                                7, 3,
        //                                8}
        //           seq_order = {1, 0, 2}, the sort order.
        //               where 1 is the second sequence,
        //                     0 is the first sequence,
        //                     2 is the third sequence.
        // The num_batch represents batch size after rearranging the
        // input LodTensor. It is also the maximum length of input sequence.

        std::vector<std::vector<int>> batch_seq_meta;
        batch_seq_meta.emplace_back(std::vector<int>{0});
        batch_seq_meta.emplace_back(std::vector<int>{0});
        batch_seq_meta.emplace_back(std::vector<int>{0});

        // batch_seq_meta[0] is the start positions for batch LoDTensor
        int num_batch = seq_info[0].length;
        batch_seq_meta[0].resize(static_cast<int>(num_batch + 1));
        // batch_seq_meta[1] is the raw index in the input LoDTensor
        batch_seq_meta[1].resize(static_cast<int>(seq->num()));
        // batch_seq_meta[2] is the sort order for the input LoDTensor.
        batch_seq_meta[2].resize(seq_info.size());

        int* batch_starts = batch_seq_meta[0].data();
        int* seq2batch_idx = batch_seq_meta[1].data();
        batch_starts[0] = 0;
        for (int n = 0; n < num_batch; n++) {
            auto batch_id = static_cast<int>(batch_starts[n]);
            for (int i = 0; i < seq_info.size(); ++i) {
                int seq_len = seq_info[i].length;
                int start = seq_info[i].start;
                if (n < seq_len) {
                    seq2batch_idx[batch_id] =
                        is_reverse ? start + seq_len - 1 - n : start + n;
                    batch_id++;
                } else {
                    break;
                }
            }
            batch_starts[n + 1] = static_cast<int>(batch_id);
        }
        int* seq_order = batch_seq_meta[2].data();
        for (int i = 0; i < seq_info.size(); ++i) {
            seq_order[i] = seq_info[i].seq_idx;
        }
        seq_to_batch_meta = batch_seq_meta;

        CopyMatrixRowsFunctor<Dtype, LayOutType> to_batch;
        to_batch(seq, batch_seq_meta[1], batch, true);
    }
};

template <DataType Dtype, typename LayOutType>
class Batch2LoDTensorFunctor {
public:
    typedef Tensor<X86, Dtype, LayOutType> ioTensor;
    void operator()(ioTensor* batch,
                  ioTensor* seq, std::vector<std::vector<int>>& seq_to_batch_meta) const {
        if (seq_to_batch_meta.size() < 2) {
            LOG(ERROR) << "The size of seq_to_batch_meta should inlcude at least 2-level sequence information.";
            exit(-1);
        }
        if (seq_to_batch_meta[1].size() != static_cast<int>(seq->num())) {
            LOG(ERROR) << "The seq_to_batch information should be consistent with the dims.";
            exit(-1);
        }
        CopyMatrixRowsFunctor<Dtype, LayOutType> to_seq;
        to_seq(batch, seq_to_batch_meta[1], seq, false);
    }
};

template <DataType Dtype, typename LayOutType>
class ReorderInitState {
public:
    typedef Tensor<X86, Dtype, LayOutType> ioTensor;
    void operator()(ioTensor *src, std::vector<int> ind_lod, ioTensor *dst, bool indexed_src) {
        math::CopyMatrixRowsFunctor<Dtype, LayOutType> row_shuffle;
        row_shuffle(src, ind_lod, dst, indexed_src);
    }
};
}  // namespace math
}  // namespace saber
}  // namespace anakin

#endif
