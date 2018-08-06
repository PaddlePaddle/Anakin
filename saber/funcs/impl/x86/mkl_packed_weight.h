#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_MKL_PACKED_WEIGHT_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_MKL_PACKED_WEIGHT_H

#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <mkl_vml_functions.h>
#include "saber/core/tensor.h"

namespace anakin {
namespace saber {

template <typename T>
class MatrixInfo {
public:
    // default construct
    MatrixInfo() :
        _buf_(nullptr), _height_(0),  _width_(0) {
    };

    // construct the class with allocated buf and
    // the matrix info including height(row number) and width(column number)
    MatrixInfo(T *buf, size_t height, size_t width) :
        _buf_(buf), _height_(height),  _width_(width){
    };

    // get the raw data buf point
    T *buf() {
        return _buf_;
    };

    // return the height of the buffer;
    // equal to row number
    size_t height() {
        return _height_;
    };

    // return the width of the buffer;
    // equal to column number
    size_t width() {
        return  _width_;
    }

    // get the sub buf between start and end;
    // return sub buffer : [start,end)
    MatrixInfo<T> subMatrixInfo(int start, int end) {
        MatrixInfo<T> ret(_buf_ + start * _width_, end -start, _width_);
        return ret;
    }

    // print the value to log
    void log_dump() {
        for (int i = 0;  i < _height_ * _width_; i++) {
            LOG(INFO) <<"i:" <<i << " value:" <<*(_buf_ + i);
        }
    }

    // clean the buffer with zero
    void zero() {
        memset(_buf_, 0,_height_ * _width_ * sizeof(T));
    }

private:
    T *_buf_;
    size_t _height_;
    size_t  _width_;
};

template <DataType Dtype, typename LayOutType>
class mkl_packed_weight {

public:
    typedef Tensor<X86, Dtype, LayOutType> ioTensor;
    typedef typename ioTensor::Dtype dtype;
    explicit mkl_packed_weight(MatrixInfo<dtype> *weight, bool transW = false) {
        packed_weight_ = nullptr;
        weight_ = weight->buf();
        height_ = weight->height();
        width_ = weight->width();
        trans_w_ = transW;
    }

    ~mkl_packed_weight() {
        if (packed_weight_) {
            cblas_sgemm_free(packed_weight_);
            packed_weight_ = nullptr;
        }
    }

    void pack() {
        if (!packed_weight_) {
            packed_weight_ = cblas_sgemm_alloc(CblasBMatrix, 1, width_, height_);
        }
        cblas_sgemm_pack(CblasRowMajor,
                     CblasBMatrix,
                     CblasNoTrans,
                     1,
                     width_,
                     height_,
                     1.0,
                     weight_,
                     width_,
                     packed_weight_);
    }

    void gemm_compute(MatrixInfo<dtype>& src, MatrixInfo<dtype>* dst, float beta = 1.0) {
        cblas_sgemm_compute(CblasRowMajor,
                        CblasNoTrans,
                        CblasPacked,
                        src.height(),
                        width_,
                        height_,
                        src.buf(),
                        src.width(),
                        packed_weight_,
                        width_,
                        beta,
                        dst->buf(),
                        dst->width()
                        );
    }
protected:
    /// The pointer of weight
    dtype * weight_;
    /// The pointer of cblas packed gemm to weight
    dtype *packed_weight_;
    size_t height_;
    size_t width_;
    bool trans_w_;
};

template class mkl_packed_weight<AK_FLOAT, NCHW>;

}  // namespace saber
}  // namespace anakin

#endif
