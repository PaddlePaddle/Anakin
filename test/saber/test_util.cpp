#include "test_util.h"

int main(){
    VectorEX<float> a({1,2,3});
    VectorEX<float> b({1,2,3});
//    printf("what %d\n",a.size());
    std::cout<<a/a+b*a<<std::endl;
    std::cout<<a-b<<std::endl;
    std::cout<<a*b<<std::endl;
    std::cout<<a/b<<std::endl;
    for(ActiveType type:{Active_sigmoid,Active_tanh,Active_relu,Active_identity})
        std::cout<<active_vectorEx(a,type)<<std::endl;

}