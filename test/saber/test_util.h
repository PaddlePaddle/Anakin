
#ifndef ANAKIN_TEST_SABER_TEST_UTIL_H
#define ANAKIN_TEST_SABER_TEST_UTIL_H

#include "vector"
#include "iostream"
#include "saber_types.h"
#include "math.h"
using namespace anakin::saber;





template<typename Dtype,typename ...Args>
class VectorEX : std::vector<Dtype>{
public:
    using vector = std::vector<Dtype>;

    VectorEX(vector data):vector(data){};
    VectorEX(int size):vector(size){};
    VectorEX(Dtype *data,int length){
        for(int i=0;i<length;i++){
            push_back(data[i]);
        }
    }
    int size(){
        return vector::size();
    }

    Dtype* data(){
        return vector::data();
    }

    VectorEX& operator =(const VectorEX& right){
        this->clear();
        for (int i = 0; i < right.size(); ++i) {
            this->push_back(right[i]);
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& out, const VectorEX& s)
    {

        out<<"[";
        for(int i=0;i<s.size();++i){
            out<<s[i]<<",";
        }
        out<<"]";
        return out;
    }

    void fill_to_ptr(Dtype* target){
        for(int i=0;i<size();++i){
            target[i]=data()[i];
        }
    }

#define VEC_OP(OP)\
    VectorEX operator OP(VectorEX b){\
        VectorEX c(b.size());\
        for(int i=0;i<b.size();++i){\
            c[i]=data()[i] OP b[i];\
        }\
        return c;\
    }

    VEC_OP(+);
    VEC_OP(-);
    VEC_OP(*);
    VEC_OP(/);



    Dtype InValidAct(Dtype a) {
        CHECK(false)<<"InValidAct";
    }

    Dtype Sigmoid(const Dtype a) {
        return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-a));
    }

    Dtype Tanh(const Dtype a) {
        Dtype tmp = -2.0 * a;
        return (2.0 / (1.0 + exp(tmp))) - 1.0;
    }

    Dtype Relu(const Dtype a) {
        return a > static_cast<Dtype>(0.0) ? a : static_cast<Dtype>(0.0);
    }

    Dtype Identity(const Dtype a) {
        return a;
    }

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0

     Dtype Sigmoid_fluid(const Dtype a) {
        const Dtype min = SIGMOID_THRESHOLD_MIN;
        const Dtype max = SIGMOID_THRESHOLD_MAX;
        Dtype tmp = (a < min) ? min : ((a > max) ? max : a);
        return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-tmp));
    }

     Dtype Tanh_fluid(const Dtype a) {
        Dtype tmp = -2.0 * a;
        tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
        return (2.0 / (1.0 + exp(tmp))) - 1.0;
    }

    void map(void* func){
        Dtype(*act)(const Dtype)=func;
        for(int i=0;i<size();++i){
            data()[i]=act(i);
        }
    }

    VectorEX& active(ActiveType type){
        switch (type){
            case Active_sigmoid:map((void*)&Sigmoid);
                break;
            case Active_sigmoid_fluid:map((void*)&Sigmoid_fluid);
                break;
            case Active_tanh:map((void*)&Tanh);
                break;
            case Active_tanh_fluid:map((void*)&Tanh_fluid);
                break;
            case Active_relu:map((void*)&Relu);
                break;
            case Active_identity:map((void*)&Identity);
                break;
            default:map((void*)&Identity);
        }

        return *this;
    }

};
template <typename Dtype>
VectorEX<Dtype>& active_vectorEx(VectorEX<Dtype>& vex,ActiveType type){
    return vex.active(type);
}

#endif //ANAKIN_TEST_SABER_TEST_UTIL_H
