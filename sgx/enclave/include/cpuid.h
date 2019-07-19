#ifndef ANAKIN_SGX_CPUID_H
#define ANAKIN_SGX_CPUID_H

#ifdef SGX_I7
#include "cpuid_i7.h"
#else
#include "cpuid_e3.h"
#endif

#endif
