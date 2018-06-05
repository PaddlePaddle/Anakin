#ifndef MERCURY_BASE_SHAPE_H
#define MERCURY_BASE_SHAPE_H
#include<algorithm>
#include <vector>
#include "base/common.h"

namespace mercury{

//! default layout is NCHW, CHW, HW, W
//! maximum dim is 4

class Shape : public std::vector<int> {
public:
    using vector = std::vector<int>;

    Shape():vector(){}
    template <typename First, typename ...Args>
    Shape(First first, Args... res) {
       init_dims(first, res...); 
    }

    int num() const {
        if (dims() == 0) {
            return 0;
        }
        int i = 1;
        if (dims() == 4) {
            i =  data()[0];
        }
        return i;
    }
    int channel() const {
        if (dims() == 0) {
            return 0;
        }
        int i = 1;
        if (dims() >= 3) {
            i =  data()[dims() - 3];
        }
        return i;
    }

    int height() const {
        if (dims() == 0) {
            return 0;
        }
        int i = 1;
        if (dims() >= 2) {
            i =  data()[dims() - 2];
        }
        return i;
    }

    int width() const {
        if (dims() == 0) {
            return 0;
        }
        return data()[dims() - 1];
    }

    Shape stride() const {
        CHECK_GT(dims(), 0) << "shape is empty";
        Shape sh(dims());
        for (int i = 0; i < dims(); ++i) {
            sh[i] = count(i + 1);
        }
        return sh;
    }

    void set_num(int num) {
        CHECK_GT(dims(), 0) << "shape is empty";
        if (dims() == 4) {
            data()[0] = num;
        }
    }

    void set_channel(int channel) {
        CHECK_GT(dims(), 0) << "shape is empty";
        if (dims() >= 3) {
            data()[dims() - 3] = channel;
        }
    }

    void set_height(int height) {
        CHECK_GT(dims(), 0) << "shape is empty";
        if (dims() >= 2) {
            data()[dims() - 2] = height;
        }
    }

    void set_width(int width) {
        CHECK_GT(dims(), 0) << "shape is empty";
        if (dims() >= 1) {
            data()[dims() - 1] = width;
        }
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

    int count(int start = 0) const {
        if (this->size() == 0) {
            return 0;
        }
        int sum = 1;
        for_each(this->begin()+start, this->end(), [&](int n){sum *= n;});
        return sum;
    }

    int count(int start, int end) {
        int dim = dims();
        if (start > end || start > dim) {
            return 0;
        }
        if (start < 0) {
            start = 0;
        }
        if (end > dim) {
            end = dim;
        }
        int sum = 1;
        for (int i = start; i < end; ++i) {
            sum *= this->data()[i];
        }
        return sum;
    }

    int dims() const {
        return this->size();
    }

    static Shape zero(int dims){
        Shape sh;
        for (int i = 0; i < dims; ++i) {
            sh.push_back(0);
        }
        return sh;
    }

    static Shape minusone(int dims){
        Shape sh;
        for (int i = 0; i < dims; ++i) {
            sh.push_back(-1);
        }
        return sh;
    }

private:
    template <typename First, typename ...Args>
    void init_dims(First head, Args...args){
        push_back(head);
        init_dims(args...);
    }
    void init_dims(){};
};

} //namespace mercury

#endif //MERCURY_BASE_SHAPE_H
