#ifndef ANAKIN_SABER_FUNC_IMPL_X86_MATH_SEQUENCE_BATCH_H
#define ANAKIN_SABER_FUNC_IMPL_X86_MATH_SEQUENCE_BATCH_H

#include <algorithm>
#include <vector>
#include "saber/core/tensor.h"
#ifdef USE_OPENMP
#include "omp.h"
#endif
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
                    bool is_src_index, int fragment_num);
};

template <DataType Dtype, typename LayOutType>
class Seq2BatchFunctor {
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
                    bool is_reverse = false, int fragment_num = 1) const {
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
            to_batch(seq, seq_to_batch_meta[1], batch, true, fragment_num);
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
        [](SeqInfo a, SeqInfo b) {
            return a.length > b.length;
        });

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
        batch_seq_meta.emplace_back(std::vector<int> {0});
        batch_seq_meta.emplace_back(std::vector<int> {0});
        batch_seq_meta.emplace_back(std::vector<int> {0});

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
        to_batch(seq, batch_seq_meta[1], batch, true, fragment_num);
    }
};

template <DataType Dtype, typename LayOutType>
class Batch2SeqFunctor {
public:
    typedef Tensor<X86, Dtype, LayOutType> ioTensor;
    void operator()(ioTensor* batch,
                    ioTensor* seq, std::vector<std::vector<int>>& seq_to_batch_meta, int fragment_num = 1) const {
        if (seq_to_batch_meta.size() < 2) {
            LOG(ERROR) << "The size of seq_to_batch_meta should inlcude at least 2-level sequence information.";
            exit(-1);
        }

        if (seq_to_batch_meta[1].size() != static_cast<int>(seq->num())) {
            LOG(ERROR) << "The seq_to_batch information should be consistent with the dims.";
            exit(-1);
        }

        CopyMatrixRowsFunctor<Dtype, LayOutType> to_seq;
        to_seq(batch, seq_to_batch_meta[1], seq, false, fragment_num);
    }
};

template <DataType Dtype, typename LayOutType>
class ReorderInitState {
public:
    typedef Tensor<X86, Dtype, LayOutType> ioTensor;
    void operator()(ioTensor* src, std::vector<int> ind_lod, ioTensor* dst, bool indexed_src,
                    int fragment_num = 1) {
        math::CopyMatrixRowsFunctor<Dtype, LayOutType> row_shuffle;
        row_shuffle(src, ind_lod, dst, indexed_src, fragment_num);
    }
};


/*
 * This class can used to modify the matrix structure of sequence matrix into
 * batch structure.
 * sequence matrix: [C1_s ... Cn_s | ...... | C1_t ... Cn_t]
 * batch matrix:    [C1_s ... C1_t | ...... | Cn_s ... Cn_t]
 * Cn_s is the state for sequence s at time n.
 *
 *  Exampel:  sequence matrix = {{0, 0, 0, 0}, {1, 1, 1, 1, 1}, {2, 2, 2}}
 *            s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
 *            batch matrix = {{1, 0, 2}, {1, 0, 2}, {1, 0, 2}, {1, 0}, {1}}
 *            b0: 1 0 2, b1: 1 0 2, b2: 1 0 2, b3: 1 0, b4: 1
 *
 *  Use:
 *            Input: seqMatrix, seqStarts(Sequence Start Positions)
 *            Output: batchMatrix
 *            1. SequenceToBatch seq2batch;
 *            2. seq2batch.resizeOrCreateBatch(seqStarts);     // calculate seq2BatchIdx
 *            3. seq2batch.copy(seqMatrix, batchMatrix, true); // copy seq to batch matrix
 *
 */

class SequenceToBatch {
public:
    SequenceToBatch() {};

    template <typename Dtype>
    void seq_2_bat(const Dtype*  input, Dtype* output, int word_size) {
        int word_sum = seq2BatchIdx_.size();
        #pragma omp parallel for if(thread_num > 1)

        for (int old_id = 0; old_id < word_sum; ++old_id) {
            int word_start = old_id * word_size;
            int maped_id = seq2BatchIdx_[old_id];
            int maped_start = maped_id * word_size;

            for (int word_offset = 0; word_offset < word_size; ++word_offset) {
                output[word_start + word_offset] = input[maped_start + word_offset];
            }
        }
    }

    template <typename Dtype>
    void hidden_2_bat(const Dtype* input, Dtype* output, int hidden_size) {
        int batch_size = seqStartAndLength_.size();

        for (int old_id = 0; old_id < batch_size; ++old_id) {
            int word_start = old_id * hidden_size;
            int maped_id = seqStartAndLength_[old_id].seqIdx_;
            int maped_start = maped_id * hidden_size;

            for (int word_offset = 0; word_offset < hidden_size; ++word_offset) {
                output[word_start + word_offset] = input[maped_start + word_offset];
            }
        }
    }

    template <typename Dtype>
    void bat_2_seq(const Dtype* input, Dtype* output, int hidden_size) {
        int word_sum = seq2BatchIdx_.size();
        #pragma omp parallel for if(thread_num > 1)

        for (int old_id = 0; old_id < word_sum; old_id++) {
            int word_start = old_id * hidden_size;
            int maped_id = seq2BatchIdx_[old_id];
            int maped_start = maped_id * hidden_size;

            for (int word_offset = 0; word_offset < hidden_size; word_offset++) {
                output[maped_start + word_offset] = input[word_start + word_offset];
            }
        }
    }

    template <typename Dtype>
    void bat_2_seq(const Dtype* input, Dtype* output, int hidden_size, int aligned_hidden_size) {
        int word_sum = seq2BatchIdx_.size();
        #pragma omp parallel for if(thread_num > 1)

        for (int old_id = 0; old_id < word_sum; old_id++) {
            int word_start = old_id * aligned_hidden_size;
            int maped_id = seq2BatchIdx_[old_id];
            int maped_start = maped_id * hidden_size;

            for (int word_offset = 0; word_offset < hidden_size; word_offset++) {
                output[maped_start + word_offset] = input[word_start + word_offset];
            }
        }
    }

    void get_batch_offset(std::vector<int>& bat_offset) {
        for (size_t i = 0; i < batchStartPositions_.size(); i++) {
            bat_offset[i] = batchStartPositions_[i];
        }
    }

    size_t get_batch_num() const {
        return numBatch_;
    }

    void create_batch(int batchSize, size_t numSequences, std::vector<int>& seqStarts,
                      bool reversed) {
        CHECK_EQ(seqStarts[numSequences], batchSize);
        seq2BatchIdx_.resize(batchSize);

        /*
         * calculate the length of each sequence & sort sequence index by the length
         * Exampel:  Sequences = {s0, s1, s2}
         *           s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
         *           seqStartAndLength_[3] = {(4, 5, 1), (0, 4, 0), (9, 3, 2)}
         */
        for (size_t seqId = 0; seqId < numSequences; ++seqId) {
            int length = seqStarts[seqId + 1] - seqStarts[seqId];
            seqStartAndLength_.emplace_back(seqStarts[seqId], length, seqId);
        }

        std::sort(seqStartAndLength_.begin(), seqStartAndLength_.end(),
        [](SeqStartAndLength a, SeqStartAndLength b) {
            return a.length_ > b.length_;
        });

        /*
         * calculate the start position of each batch
         * (numBatch equal the maxLength of sequences)
         * Exampel:  Sequences = {s0, s1, s2}
         *           s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
         *           numBatch = 5,
         *           batchIndex = {b0, b1, b2, b3, b4}
         *           b0: 1 0 2, b1: 1 0 2, b2: 1 0 2, b3: 1 0, b4: 1
         *           batchStartPositions[6] = {0, 3, 6, 9, 11, 12}
         */
        numBatch_ = (size_t)seqStartAndLength_[0].length_;
        batchStartPositions_.resize(numBatch_ + 1);
        batchStartPositions_[0] = 0;

        for (size_t n = 0; n < numBatch_; n++) {
            int batchId = batchStartPositions_[n];

            for (size_t i = 0; i < seqStartAndLength_.size(); ++i) {
                size_t seqLength = seqStartAndLength_[i].length_;
                int start = seqStartAndLength_[i].start_;

                if (n < seqLength) {
                    if (!reversed) {
                        seq2BatchIdx_[batchId] = start + n;
                    } else {
                        seq2BatchIdx_[batchId] = start + seqLength - 1 - n;
                    }

                    batchId++;
                } else {
                    break;
                }
            }

            batchStartPositions_[n + 1] = batchId;
        }
    }


protected:
    struct SeqStartAndLength {
        int start_;
        int length_;
        int seqIdx_;
        SeqStartAndLength(int start, int length, int seqIdx)
            : start_(start), length_(length), seqIdx_(seqIdx) {}
    };
    std::vector<SeqStartAndLength> seqStartAndLength_;
    std::vector<int> batchStartPositions_;
    std::vector<int> seq2BatchIdx_;
    size_t numBatch_;
#ifdef USE_OPENMP
    int thread_num = omp_get_max_threads();
#endif
};
}  // namespace math
}  // namespace saber
}  // namespace anakin

#endif
