/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef ANAKIN_SABER_CORE_SHAPE_H
#define ANAKIN_SABER_CORE_SHAPE_H

#include <vector>
#include "core/common.h"

namespace anakin{

namespace saber{

class Shape : public std::vector<int> {
public:
    using vector = std::vector<int>;

    Shape() : vector(), _layout(nullptr) {}

    Shape(vector data, LayoutType layout_type = Layout_NCHW) {
        create_layout(layout_type);
        CHECK_EQ(_layout->dims(), data.size());
        for (int i = 0; i < _layout->dims(); ++i) {
            this->push_back(data[i]);
        }
        if (_layout->inner_c() != -1) {
            CHECK_EQ(data[4], _layout->inner_c()) \
                << " Layout must be an integer multiple of "
                << _layout->inner_c();
        }
    }
    ~Shape() {
        delete _layout;
        _layout = nullptr;
    }

    Shape(const Shape& right) {
        this->clear();
        for (int i = 0; i < right.size(); ++i) {
            this->push_back(right[i]);
        }
        create_layout(right.get_layout());
    }

    Shape &operator=(const Shape& right) {
        this->clear();
        for (int i = 0; i < right.size(); ++i) {
            this->push_back(right[i]);
        }
        delete _layout;
        _layout = nullptr;
        create_layout(right.get_layout());
        return *this;
    }
    Shape operator+(const Shape& shape) {

        Shape tmp_shape(*this);
        int* p = data();
        for (size_t i = 0; i < size(); i++) {
            tmp_shape[i] = p[i] + shape[i];
        }
        return tmp_shape;
    }

    Shape operator-(const Shape& shape) {

        Shape tmp_shape(*this);
        int* p = data();
        for (size_t i = 0; i < size(); i++) {
            tmp_shape[i] = p[i] - shape[i];
        }
        return tmp_shape;  
    }

    bool operator<(const Shape& shape) const {

        bool flag = size() == shape.size();
        if (!flag) {
            return false;
        }

        const int* p = data();
        for (size_t i = 0; i < size(); i++) {
            flag &= (p[i] < shape[i]);
        }
        return flag;
    }

    bool operator<=(const Shape& shape) const{

        bool flag = size() == shape.size();
        if (!flag) {
            return false;
        }
        const int* p = data();
        for (size_t i = 0; i < size(); i++) {
            flag &= (p[i] <= shape[i]);
        }
        return flag;
    }

    bool operator>(const Shape& shape) const {

        bool flag = size() == shape.size();
        if (!flag) {
            return false;
        }

        const int* p = data();
        for (size_t i = 0; i > size(); i++) {
            flag &= (p[i] > shape[i]);
        }
        return flag;
    }

    bool operator>=(const Shape& shape) const{

        bool flag = size() == shape.size();
        if (!flag) {
            return false;
        }
        const int* p = data();
        for (size_t i = 0; i > size(); i++) {
            flag &= (p[i] >= shape[i]);
        }
        return flag;
    }

    bool operator==(const Shape& shape) const{

        bool flag = size() == shape.size();
        if (!flag) {
            return false;
        }
        const int* p = data();
        for (size_t i = 0; i < size(); i++) {
            flag &= (p[i] == shape[i]);
        }
        return flag;
    }
    int num_index() const {
        return _layout->num_index();
    }
    int channel_index() const {
        return _layout->channel_index();
    }
    int height_index() const {
        return _layout->height_index();
    }
    int width_index() const {
        return _layout->width_index();
    }
    int depth_index() const {
        return _layout->depth_index();
    }
    int num() const {
        int shape_num = this->num_index() == -1 ? 1 : this->data()[this->num_index()];
        return shape_num;
    }
    int channel() const {
        int shape_channel = this->channel_index() == -1 ? 1 : this->data()[this->channel_index()];
        if (_layout->inner_c() != -1) {
            shape_channel *= _layout->inner_c();
        }
        return shape_channel;
    }
    int height() const {
        int shape_height = this->height_index() == -1 ? 1 : this->data()[this->height_index()];
        return shape_height;
    }
    int width() const {
        int shape_width = this->width_index() == -1 ? 1 : this->data()[this->width_index()];
        return shape_width;
    }
    int depth() const {
        int shape_depth = this->depth_index() == -1 ? 1 : this->data()[this->depth_index()];
        return shape_depth;
    }
    long long count(int start = 0) const {
        if (start > dims()) {
            start = dims();
        }
        if (this->size() == 0) {
            return 0;
        }
        long long sum = 1;
        for_each(this->begin() + start, this->end(), [&](int n){sum *= n;});
        return sum;
    }
    long long count(int start, int end) const {
        if (start < 0) {
            start = 0;
        }
        if (end > dims()) {
            end = dims();
        }
        if (end < start) {
            end = start;
        }
        long long  sum  = 1;
        for (int i = start; i < end; ++i) {
            sum *= data()[i];
        }
        return sum;
    }
    Shape get_stride() {
        Shape data_stride = Shape::zero(*this);
        for (int i = 0; i < dims(); ++i) {
            data_stride[i] = count(i + 1);
        }
        return data_stride;
    }
    int dims() const {
        return this->size();
    }

    bool is_continue(const Shape real_shape) const {
        if (real_shape.size() != this->size()){
            return false;
        }

        const int* p = data();
        for (int i = this->size() - 1; i >= 0; i--) {
            if (p[i] != real_shape[i]) {
                int size = this->count() / this->count(i);
                return size == 1;
            }
        }
        return true;
    }
    LayoutType get_layout() const {
        return _layout->type();
    }
    static Shape zero(const Shape &right){
        Shape sh = right;
        for (int i = 0; i < right.size(); ++i) {
            sh[i] = 0;
        }
        return sh;
    }

    static Shape minusone(const Shape &right){
        Shape sh = right;
        for (int i = 0; i < right.size(); ++i) {
            sh[i] = -1;
        }
        return sh;
    }

protected:
    Layout* _layout{nullptr};
private:
    void create_layout(LayoutType layout_type) {
        switch(layout_type) {
            case Layout_invalid: this->_layout = nullptr; break;
            case Layout_W: this->_layout = new W(); break;
            case Layout_HW: this->_layout = new HW(); break;
            case Layout_WH: this->_layout = new WH(); break;
            case Layout_NW: this->_layout = new NW(); break;
            case Layout_NHW: this->_layout = new NHW(); break;
            case Layout_NCHW: this->_layout = new NCHW(); break;
            case Layout_NHWC: this->_layout = new NHWC(); break;
            case Layout_NCHW_C4: this->_layout = new NCHW_C4(); break;
            case Layout_NCHW_C8: this->_layout = new NCHW_C8(); break;
            case Layout_NCHW_C16: this->_layout = new NCHW_C16(); break;
        }
    }
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_SHAPE_H
