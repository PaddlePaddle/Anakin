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

    Shape():vector(){}
    template <typename First, typename ...Args>
    Shape(First first, Args... res) {
       init_dims(first, res...); 
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

    Shape(const Shape& right) {
        this->clear();
        for (int i = 0; i < right.size(); ++i) {
            this->push_back(right[i]);
        }
    }

    Shape &operator=(const Shape& right) {
        this->clear();
        for (int i = 0; i < right.size(); ++i) {
            this->push_back(right[i]);
        }
        return *this;
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

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_SHAPE_H
