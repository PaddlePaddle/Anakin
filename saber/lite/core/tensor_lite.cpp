#include "saber/lite/core/tensor_lite.h"

namespace anakin{

namespace saber{

namespace lite{
template <typename Dtype>
    Tensor<Dtype>::Tensor() {
        _buf = std::make_shared<CpuBuffer>();
        _target_type = eARM;
        _is_subbuf = false;
        _is_shared = false;
    }

template <typename Dtype>
Tensor<Dtype>::Tensor(Shape shape, TargetTypeEnum target) {
        _target_type = target;
        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape.dims());
        _buf = std::make_shared<CpuBuffer>(shape.count() * _type_len);
        _is_subbuf = false;
    }

    template <typename Dtype>
    Tensor<Dtype>::Tensor(Dtype* data_ptr, TargetTypeEnum target, int id, Shape shape) {

        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape.dims());
        _buf = std::make_shared<CpuBuffer>(data_ptr, shape.count() * _type_len);
        _is_shared = true;
        _is_subbuf = false;
    }

    template <typename Dtype>
    Tensor<Dtype>::Tensor(const Tensor<Dtype>& tensor){
        _shape = tensor._shape;
        _valid_shape = tensor._valid_shape;
        _offset = tensor._offset;
        _buf = tensor._buf;
        _is_subbuf = tensor._is_subbuf;
        _seq_offset = tensor._seq_offset;
    }

    template <typename Dtype>
    SaberStatus Tensor<Dtype>::set_shape(Shape valid_shape, Shape shape, Shape offset) {

        if (shape.dims() > 0) {
            CHECK_EQ(shape.dims(), valid_shape.dims()) << "shape dims must be the same";
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            CHECK_EQ(offset.dims(), valid_shape.dims()) << "shape dims must be the same";
            _offset = offset;
        }

        _valid_shape = valid_shape;

        if (!_is_subbuf) {
            if (_shape.count() <= _valid_shape.count()) {
                _shape = _valid_shape;
            }
            _offset = Shape::zero(valid_shape.dims());
        } else {
            auto shape_zero = Shape::zero(valid_shape.dims());
            if (_shape == shape_zero) {
                _shape = valid_shape;
            }
            CHECK_EQ(_valid_shape + _offset <= _shape, true) << \
                "valid_shape + offet should <= shape";
        }
        return SaberSuccess;
    }

    template <typename Dtype>
    SaberStatus Tensor<Dtype>::re_alloc(Shape shape){
        CHECK_EQ(_is_shared || _is_subbuf, false) << \
            "shared tensor could not re_alloc";
        _shape = shape;
        _valid_shape = _shape;
        _offset = Shape::zero(_shape.dims());
        _buf->alloc(_shape.count() * _type_len);
        return SaberSuccess;
    }


    template <typename Dtype>
    SaberStatus Tensor<Dtype>::reshape(Shape valid_shape, Shape shape, Shape offset) {

        if (shape.dims() > 0) {
            CHECK_EQ(shape.dims(), valid_shape.dims()) << "shape dims must be the same";
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            CHECK_EQ(offset.dims(), valid_shape.dims()) << "shape dims must be the same";
            _offset = offset;
        }

        _valid_shape = valid_shape;

        if (!_is_subbuf) {
            if (_shape.count() < _valid_shape.count()) {
                _shape = _valid_shape;
            }
            _offset = Shape::zero(valid_shape.dims());
        } else {
            CHECK_EQ(_valid_shape + _offset <= _shape, true) << \
                "valid_shape + offet should <= shape";
        }
        bool exceed_flag = _shape.count() * _type_len > _buf->get_capacity() \
            && (_is_subbuf || _is_shared);
        CHECK_EQ(exceed_flag, false) << "shared tensor shape exceed origin data buffer size";
        LITE_CHECK(_buf->re_alloc(_shape.count() * _type_len));
        return SaberSuccess;
    }

    template <typename Dtype>
    bool Tensor<Dtype>::is_continue_mem() const {
        if (!_is_subbuf) {
            return true;
        }
        return _valid_shape.is_continue(_shape);
    }

    template <typename Dtype>
    int Tensor<Dtype>::count(int start, int end) const {

        CHECK_GE(start, 0) << "start index shold >= 0!";
        CHECK_LE(end, _shape.size()) << "end index shold <= shape dims!";
        CHECK_LE(start, end) << "start index should < end index!";
        int sum  = 1;
        for (int i = start; i < end; ++i) {
            sum *= _shape[i];
        }
        return sum;
    }

    template <typename Dtype>
    int Tensor<Dtype>::count_valid(int start, int end) const {

        CHECK_GE(start, 0) << "start index shold >= 0!";
        CHECK_LE(end, _valid_shape.size()) << "end index shold <= shape dims!";
        CHECK_LE(start, end) << "start index should < end index!";
        int sum  = 1;
        for (int i = start; i < end; ++i) {
            sum *= _valid_shape[i];
        }
        return sum;
    }

    template <typename Dtype>
    int Tensor<Dtype>::size() const {
        return _shape.count();
    }

    template <typename Dtype>
    int Tensor<Dtype>::valid_size() const{
        return _valid_shape.count();
    }

    template <typename Dtype>
    int Tensor<Dtype>::dims() const {
        return _valid_shape.dims();
    }

    template <typename Dtype>
    Shape Tensor<Dtype>::shape() const{
        return _shape;
    }

    template <typename Dtype>
    Shape Tensor<Dtype>::valid_shape() const {
        return _valid_shape;
    }

    template <typename Dtype>
    Shape Tensor<Dtype>::get_stride() const {
        Shape data_stride = Shape::zero(dims());
        if (_is_subbuf) {
            for (int i = 0; i < dims(); ++i) {
                data_stride[i] = _shape.count(i + 1);
            }
        } else {
            for (int i = 0; i < dims(); ++i) {
                data_stride[i] = _valid_shape.count(i + 1);
            }
        }

        return data_stride;
    }

    template <typename Dtype>
    Shape Tensor<Dtype>::offset() const {
        return _offset;
    }

    template <typename Dtype>
    int Tensor<Dtype>::device_id() const {
        return 0;
    }

    template <typename Dtype>
    int Tensor<Dtype>::num() const {
        return _valid_shape.num();
    }

    template <typename Dtype>
    void Tensor<Dtype>::set_num(int num) {
        return _valid_shape.set_num(num);
    };

    template <typename Dtype>
    int Tensor<Dtype>::channel() const {
        return _valid_shape.channel();
    }

    template <typename Dtype>
    void Tensor<Dtype>::set_channel(int channel) {
        return _valid_shape.set_channel(channel);
    }

    template <typename Dtype>
    int Tensor<Dtype>::height() const {
        return _valid_shape.height();
    }

    template <typename Dtype>
    void Tensor<Dtype>::set_height(int h) {
        return _valid_shape.set_height(h);
    }

    template <typename Dtype>
    int Tensor<Dtype>::width() const {
        return _valid_shape.width();
    }

    template <typename Dtype>
    void Tensor<Dtype>::set_width(int w) {
        return _valid_shape.set_width(w);
    }

    template <typename Dtype>
    Dtype* Tensor<Dtype>::mutable_data(int index) {
        if (_buf->get_capacity() == 0){
            return nullptr;
        }
        return static_cast<Dtype*>(_buf->get_data_mutable()) + start_index() + index;
    }

    template <typename Dtype>
    const Dtype * Tensor<Dtype>::data(int index) const {
        if (_buf->get_capacity() == 0){
            return nullptr;
        }
        return static_cast<const Dtype*>(_buf->get_data()) + start_index() + index;
    }

    template <typename Dtype>
    const std::shared_ptr<Buffer>& Tensor<Dtype>::get_buf() const {
        return _buf;
    }

    template <typename Dtype>
    template <typename Tensor_t>
    SaberStatus Tensor<Dtype>::share_from(const Tensor_t& tensor) {

        CHECK_EQ(_shape.dims() > 0, true) << \
            "current tensor is not initialized (no shape info, use set_shape)";
        typedef typename Tensor_t::Dtype dtype_t;
        CHECK_LE(size() * _type_len, tensor.size() * sizeof(dtype_t)) << \
            "current tensor size should <= input tensor size";
        //! fixme, when use cl memory
        _buf = std::dynamic_pointer_cast<CpuBuffer>(tensor.get_buf());
        _is_shared = true;
        _is_subbuf = false;
        _seq_offset = tensor.get_seq_offset();
        return SaberSuccess;
    }

    template <typename Dtype>
    std::vector<int> Tensor<Dtype>::get_seq_offset() const {return _seq_offset;}
    template <typename Dtype>
    SaberStatus Tensor<Dtype>::set_seq_offset(std::vector<int> seq_offset) {_seq_offset = seq_offset; return SaberSuccess;}

    template <typename Dtype>
    SaberStatus Tensor<Dtype>::share_sub_buffer(const Tensor<Dtype>& tensor, \
        Shape valid_shape, Shape offset) {

        CHECK_EQ(true, (offset + valid_shape) <= tensor.shape()) << \
            "offset + valid_shape <= shape";
        _valid_shape = valid_shape;
        _offset = offset;
        _shape = tensor.shape();
        _buf = std::dynamic_pointer_cast<CpuBuffer>(tensor.get_buf());//tensor.get_buf();
        _is_subbuf = true;
        _is_shared = true;
        _seq_offset = tensor.get_seq_offset();
        return SaberSuccess;
    }

    template <typename Dtype>
    SaberStatus Tensor<Dtype>::copy_from(const Tensor<Dtype>& tensor) {

        CHECK_EQ(valid_size(), tensor.valid_size()) \
            << "sizes of two valid shapes must be the same";

        std::shared_ptr<CpuBuffer> buf_tmp = std::dynamic_pointer_cast<CpuBuffer>(tensor.get_buf());
        _buf->copy_from(*buf_tmp);

        return SaberSuccess;
    }

    template <typename Dtype>
    void Tensor<Dtype>::sync() {
        //!fixme
    }

    template <typename Dtype>
    void Tensor<Dtype>::record_event(void* stream) {
        //! fixme
    }

template class Tensor<float>;

} //namespace lite

} //namespace saber

} //namespace anakin

