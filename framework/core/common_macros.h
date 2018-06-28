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

#ifndef ANAKIN_COMMON_MACROS_H
#define ANAKIN_COMMON_MACROS_H 

#if defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__
#  define AK_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) && (__GNUC__ >= 4)
#  define AK_EXPORT __attribute__ ((visibility ("default")))
#else
#  define AK_EXPORT
#endif

#if defined(__GNUC__) || defined(__clang__)
#    define AK_DEPRECATED __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#    define AK_DEPRECATED __declspec(deprecated)
#else
#    define AK_DEPRECATED
#endif

#if defined(__GNUC__) 
#    define AK_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER) && (_MSC_VER >= 1300)
#    define AK_NORETURN __declspec(noreturn)
#else
#    define AK_NORETURN /* nothing by default */
#endif

#if defined(__GNUC__)
#  define AK_ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined _MSC_VER
#  define AK_ALIGNED(x) __declspec(align(x))
#else
#  define AK_ALIGNED(x)
#endif

#if defined(__GNUC__)
#  define AK_NO_NULL(...) __attribute__ ((nonnull(__VA_ARGS__)))
#else
#  define AK_NO_NULL(...)
#endif

#if defined(__GNUC__)
#   define AK_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#   define AK_ATTRIBUTE_UNUSED
#endif

#if defined(_WIN32) || (defined(__APPLE__) && !TARGET_OS_IPHONE) || defined(__linux__)
#   if defined(__APPLE__)
#       define AK_THREAD_LOCAL __thread
#   else
#       define AK_THREAD_LOCAL thread_local
#   endif
#else
#   error " Anakin can't detect thread local storage."
#   define AK_THREAD_LOCAL
#endif

#define AK_CONCAT_IMPL(_a,_b) _a##_b

#define AK_CONCAT(_a,_b) AK_CONCAT_IMPL(_a, _b)

#define AK_UNIQ_NAME(_name) AK_CONCAT(_name, __COUNTER__)

#define AK_MAKE_UNIQ_OPERATOR_NAME(_name) AK_UNIQ_NAME(_name)

#endif
