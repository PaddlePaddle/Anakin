#include "anakin_cpu_arch.h"
#include <iostream>
int main(){
    CPUID_Helper _cpu_helper;
    std::cout<<_cpu_helper.get_cpu_arch();
}