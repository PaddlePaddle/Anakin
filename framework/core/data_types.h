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

#ifndef ANAKIN_DATA_TYPES_H
#define ANAKIN_DATA_TYPES_H 

#include "framework/core/parameter.h"
#include <cstddef>

namespace anakin {

template<DataType Dtype>
struct DataTypeWarpper {
    typedef void type;
};

/// Saber type map to base type.
#define SABER_TO_BASE_TYPE(Dtype, Type) \
    template<>                          \
    struct DataTypeWarpper<Dtype> {     \
        typedef Type type;              \
    }

SABER_TO_BASE_TYPE(AK_HALF, unsigned short); /// need to be aligned
SABER_TO_BASE_TYPE(AK_FLOAT, float);
SABER_TO_BASE_TYPE(AK_DOUBLE, double);
SABER_TO_BASE_TYPE(AK_INT8, int8_t);
SABER_TO_BASE_TYPE(AK_INT16, int16_t);
SABER_TO_BASE_TYPE(AK_INT32, int32_t);
SABER_TO_BASE_TYPE(AK_INT64, int64_t);
SABER_TO_BASE_TYPE(AK_UINT8, uint8_t);
SABER_TO_BASE_TYPE(AK_UINT16, uint16_t);
SABER_TO_BASE_TYPE(AK_UINT32, uint32_t);
SABER_TO_BASE_TYPE(AK_BOOL, bool);
SABER_TO_BASE_TYPE(AK_STRING, std::string);

template<typename T>
struct DataTypeRecover {
    const static DataType type = AK_FLOAT;
};

/// Base type maps to the saber's one.
#define BASE_TYPE_TO_SABER(Type, Dtype)     \
    template<>                              \
    struct DataTypeRecover<Type> {          \
        const static DataType type = Dtype; \
    }

BASE_TYPE_TO_SABER(unsigned short, AK_HALF);
BASE_TYPE_TO_SABER(float, AK_FLOAT);
BASE_TYPE_TO_SABER(double, AK_DOUBLE);
BASE_TYPE_TO_SABER(int8_t, AK_INT8);
BASE_TYPE_TO_SABER(int16_t, AK_INT16);
BASE_TYPE_TO_SABER(int, AK_INT32);
BASE_TYPE_TO_SABER(int64_t, AK_INT64);
BASE_TYPE_TO_SABER(uint8_t, AK_UINT8);
BASE_TYPE_TO_SABER(uint32_t, AK_UINT32);
BASE_TYPE_TO_SABER(bool, AK_BOOL);
BASE_TYPE_TO_SABER(std::string, AK_STRING);

template<typename T>
struct TypeWarpper {
    typedef T type;
    const std::string type_str;
};
/// Base type maps to anakin type.
#define ANAKIN_TO_TYPE_ID(typeName, typeStrName)                \
    template<>                                                  \
    struct TypeWarpper<typeName> {\
        typedef typeName type;\
        const std::string type_str = std::string(#typeStrName); \
    };

ANAKIN_TO_TYPE_ID(void, anakin_void)
ANAKIN_TO_TYPE_ID(float, anakin_float)
ANAKIN_TO_TYPE_ID(double, anakin_double)
ANAKIN_TO_TYPE_ID(unsigned short, anakin_uint16)
ANAKIN_TO_TYPE_ID(short, anakin_int16)
ANAKIN_TO_TYPE_ID(signed char, anakin_int8)
ANAKIN_TO_TYPE_ID(unsigned char, ankin_uint8)
ANAKIN_TO_TYPE_ID(int, anakin_int32)
ANAKIN_TO_TYPE_ID(unsigned int, anakin_uint32)
ANAKIN_TO_TYPE_ID(long long, anakin_int64)
ANAKIN_TO_TYPE_ID(unsigned long long, anakin_uint64)
ANAKIN_TO_TYPE_ID(bool, anakin_bool)
ANAKIN_TO_TYPE_ID(std::string, anakin_string)

/// unique type tensor
/// ANAKIN_TO_TYPE_ID(tensor, anakin_tensor)
/// unique type shape
/// ANAKIN_TO_TYPE_ID(shape, anakin_shape)
ANAKIN_TO_TYPE_ID(PTuple<void>, anakin_tuple_void)
ANAKIN_TO_TYPE_ID(PTuple<float>, anakin_tuple_float)
ANAKIN_TO_TYPE_ID(PTuple<double>, anakin_tuple_double)
ANAKIN_TO_TYPE_ID(PTuple<short>, anakin_tuple_short)
ANAKIN_TO_TYPE_ID(PTuple<unsigned short>, anakin_tuple_unsigned_short)
ANAKIN_TO_TYPE_ID(PTuple<char>, anakin_tuple_signed_char)
ANAKIN_TO_TYPE_ID(PTuple<unsigned char>, anakin_tuple_unsigned_char)
ANAKIN_TO_TYPE_ID(PTuple<int>, anakin_tuple_int)
ANAKIN_TO_TYPE_ID(PTuple<PTuple<int>>, anakin_tuple_tuple_int)
ANAKIN_TO_TYPE_ID(PTuple<unsigned int>, anakin_tuple_unsigned_int)
ANAKIN_TO_TYPE_ID(PTuple<long>, anakin_tuple_long)
ANAKIN_TO_TYPE_ID(PTuple<bool>, anakin_tuple_bool)

ANAKIN_TO_TYPE_ID(Enum, anakin_tuple_enum)


#define ANAKIN_PBLOCK_TO_TYPE_ID(target, type_id) \
	using __type##target = PBlock<target>;	\
	ANAKIN_TO_TYPE_ID(__type##target, type_id)

#ifdef USE_CUDA
	ANAKIN_PBLOCK_TO_TYPE_ID(NV, anakin_block)
#endif

#ifdef USE_X86_PLACE
	ANAKIN_PBLOCK_TO_TYPE_ID(X86, anakin_block)
#endif

#ifdef AMD_GPU
  ANAKIN_PBLOCK_TO_TYPE_ID(AMD, anakin_block)
#endif

#ifdef USE_ARM_PLACE
	ANAKIN_PBLOCK_TO_TYPE_ID(ARM, anakin_block)
#endif

template<typename T>
struct type_id {
    typedef T type;
    const std::string type_str = TypeWarpper<T>().type_str;
    bool operator==(std::string& operand) { return type_str == operand; }
    template<typename OtherType>
    bool operator==(type_id<OtherType>& operand){
        return this->type_info() == operand.type_info();
    }
    const std::string type_info() {
        return type_str;
    }
    const std::string operator()(){
        return this->type_info();
    }
};

} /* namespace anakin */

#endif
