/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
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
#ifndef LOGGER_H
#define LOGGER_H

#include "logger_core.h"

namespace logger {

/// logger inital api
inline void init(const char* argv0){
#ifndef LOGGER_SHUTDOWN 
    core::initial(argv0); 
#endif 
}

/// judge x if false or true
#define LOGGER_IS_FALSE(x) (__builtin_expect(x,0)) 
#define LOGGER_IS_TRUE(x)  (__builtin_expect(!!(x),1))

#define LOGGER_CHECK_SYMBOL_WARP(name, symbol)                                                        \
    template <typename T1, typename T2>                                                               \
    inline std::string* name(const char* expr, const T1& lvalue, const char* op, const T2& rvalue){    \
        if(LOGGER_IS_TRUE(lvalue symbol rvalue)){return nullptr;}                   \
        std::ostringstream ss;                                                      \
        ss<<"Check failed:"<< expr <<" ("<<lvalue<<" "<<op<<" "<<rvalue<<") ";      \
        return new std::string(ss.str());                                           \
    }                                                                               \
    inline std::string* name(const char* expr, int lvalue, const char* op, int rvalue){      \
        return name<int,int>(expr, lvalue, op, rvalue);                             \
    }                                                                               \
    inline std::string* name(const char* expr, char lvalue, const char* op, char rvalue){    \
        return name<char,char>(expr, lvalue, op, rvalue);                           \
    }                                                                               \
    inline std::string* name(const char* expr, std::string lvalue, const char* op, std::string rvalue){    \
        return name<std::string,std::string>(expr, lvalue, op, rvalue);             \
    }


LOGGER_CHECK_SYMBOL_WARP(CHECK_EQ_IMPL, ==)
LOGGER_CHECK_SYMBOL_WARP(CHECK_LE_IMPL, <=)
LOGGER_CHECK_SYMBOL_WARP(CHECK_GE_IMPL, >=)
LOGGER_CHECK_SYMBOL_WARP(CHECK_NE_IMPL, !=)
LOGGER_CHECK_SYMBOL_WARP(CHECK_LT_IMPL, <)
LOGGER_CHECK_SYMBOL_WARP(CHECK_GT_IMPL, >)

//#undef LOGGER_CHECK_SYMBOL_WARP

#define LOGGER_VLOG_IF(verbose, cond)                                                   \
                 (logger::verbose > logger::utils::sys::get_max_logger_verbose_level()  \
                 || (cond)==false) ? (void)0                                            \
                 : logger::core::voidify() & logger::core::LoggerMsg<logger::verbose>(__FILE__, __LINE__)

#define LOGGER_LOG_IF(verbose_name, cond) LOGGER_VLOG_IF(Verbose_##verbose_name, cond)
#define LOGGER_VLOG(verbose) LOGGER_LOG_IF(verbose, true)
#define LOGGER_LOG(verbose) LOGGER_VLOG(verbose)

#define LOGGER_CHECK_WITH_INFO(cond, info)                          \
                 LOGGER_IS_TRUE((cond)==true) ? (void)0             \
                 : logger::core::voidify() &                        \
                 logger::core::LoggerMsg<logger::Verbose_FATAL>(    \
                         "Check failed: " info " ", __FILE__, __LINE__)

#define LOGGER_CHECK(cond) LOGGER_CHECK_WITH_INFO(cond, #cond)
#define LOGGER_CHECK_NOTNULL(x) LOGGER_CHECK_WITH_INFO((x) != nullptr, #x" != nullptr")

#define LOGGER_CHECK_OP(func_name, expr1, op, expr2)                            \
    while(auto errStr = func_name(#expr1 " " #op " " #expr2, expr1, #op, expr2))  \
                 logger::core::LoggerMsg<logger::Verbose_FATAL>(errStr->c_str(), __FILE__, __LINE__)

#define LOGGER_CHECK_EQ(A, B) LOGGER_CHECK_OP(CHECK_EQ_IMPL, A, ==, B)
#define LOGGER_CHECK_NE(A, B) LOGGER_CHECK_OP(CHECK_NE_IMPL, A, !=, B)
#define LOGGER_CHECK_LE(A, B) LOGGER_CHECK_OP(CHECK_LE_IMPL, A, <=, B)
#define LOGGER_CHECK_LT(A, B) LOGGER_CHECK_OP(CHECK_LT_IMPL, A,  <, B)
#define LOGGER_CHECK_GE(A, B) LOGGER_CHECK_OP(CHECK_GE_IMPL, A, >=, B)
#define LOGGER_CHECK_GT(A, B) LOGGER_CHECK_OP(CHECK_GT_IMPL, A,  >, B)

} /* namespace logger */

#ifdef USE_LOGGER
    #ifdef LOGGER_SHUTDOWN
        #define VLOG_IF(verbose, cond)       LOGGER_LOG_IF(verbose, false)
        #define LOG_IF(verbose, cond)        LOGGER_LOG_IF(verbose, false)
        #define VLOG(verbose)                LOGGER_LOG_IF(verbose,false)
        #define LOG(verbose)                 LOGGER_LOG_IF(verbose,false)
        #define CHECK(cond)                  LOGGER_CHECK(true)
        #define CHECK_NOTNULL(x)             LOGGER_CHECK(true)
        #define CHECK_EQ(a, b)               LOGGER_CHECK(true) 
        #define CHECK_NE(a, b)               LOGGER_CHECK(true) 
        #define CHECK_LT(a, b)               LOGGER_CHECK(true) 
        #define CHECK_LE(a, b)               LOGGER_CHECK(true) 
        #define CHECK_GT(a, b)               LOGGER_CHECK(true) 
        #define CHECK_GE(a, b)               LOGGER_CHECK(true) 
        #define DVLOG_IF(verbose, cond)      LOGGER_LOG_IF(verbose, false)
        #define DLOG_IF(verbose, cond)       LOGGER_LOG_IF(verbose, false)
        #define DVLOG(verbose)               DVLOG_IF(verbose,false)
        #define DLOG(verbose)                DLOG_IF(verbose,false)
        #define DCHECK(cond)                 LOGGER_CHECK(true)
        #define DCHECK_NOTNULL(x)            LOGGER_CHECK(true)
        #define DCHECK_EQ(a, b)              LOGGER_CHECK(true)
        #define DCHECK_NE(a, b)              LOGGER_CHECK(true)
        #define DCHECK_LT(a, b)              LOGGER_CHECK(true)
        #define DCHECK_LE(a, b)              LOGGER_CHECK(true)
        #define DCHECK_GT(a, b)              LOGGER_CHECK(true)
        #define DCHECK_GE(a, b)              LOGGER_CHECK(true)

    #else
        #define VLOG_IF(verbose, cond)       LOGGER_LOG_IF(verbose, cond)
        #define LOG_IF(verbose, cond)        LOGGER_LOG_IF(verbose, cond)
        #define VLOG(verbose)                LOGGER_LOG_IF(verbose, true)
        #define LOG(verbose)                 LOGGER_LOG_IF(verbose, true)
        #define CHECK(cond)                  LOGGER_CHECK(true)
        #define CHECK_NOTNULL(x)             LOGGER_CHECK(true)
        #define CHECK_EQ(a, b)               LOGGER_CHECK_EQ(a, b)
        #define CHECK_NE(a, b)               LOGGER_CHECK_NE(a, b)
        #define CHECK_LT(a, b)               LOGGER_CHECK_LT(a, b)
        #define CHECK_LE(a, b)               LOGGER_CHECK_LE(a, b)
        #define CHECK_GT(a, b)               LOGGER_CHECK_GT(a, b)
        #define CHECK_GE(a, b)               LOGGER_CHECK_GE(a, b)

        #ifdef ENABLE_DEBUG
            #define DVLOG_IF(verbose, cond)       LOGGER_LOG_IF(verbose, cond)
            #define DLOG_IF(verbose, cond)        LOGGER_LOG_IF(verbose, cond)
            #define DVLOG(verbose)                LOGGER_VLOG(verbose)
            #define DLOG(verbose)                 LOGGER_LOG(verbose)
            #define DCHECK(cond)                  LOGGER_CHECK(cond)
            #define DCHECK_NOTNULL(x)             LOGGER_CHECK_NOTNULL(x)
            #define DCHECK_EQ(a, b)               LOGGER_CHECK_EQ(a, b)
            #define DCHECK_NE(a, b)               LOGGER_CHECK_NE(a, b)
            #define DCHECK_LT(a, b)               LOGGER_CHECK_LT(a, b)
            #define DCHECK_LE(a, b)               LOGGER_CHECK_LE(a, b)
            #define DCHECK_GT(a, b)               LOGGER_CHECK_GT(a, b)
            #define DCHECK_GE(a, b)               LOGGER_CHECK_GE(a, b)
        #else
            #define DVLOG_IF(verbose, cond)       LOGGER_LOG_IF(verbose, false)
            #define DLOG_IF(verbose, cond)        LOGGER_LOG_IF(verbose, false)
            #define DVLOG(verbose)                DVLOG_IF(verbose,false)
            #define DLOG(verbose)                 DLOG_IF(verbose,false)
            #define DCHECK(cond)                  LOGGER_CHECK(true)
            #define DCHECK_NOTNULL(x)             LOGGER_CHECK(true)
            #define DCHECK_EQ(a, b)               LOGGER_CHECK(true)
            #define DCHECK_NE(a, b)               LOGGER_CHECK(true)
            #define DCHECK_LT(a, b)               LOGGER_CHECK(true)
            #define DCHECK_LE(a, b)               LOGGER_CHECK(true)
            #define DCHECK_GT(a, b)               LOGGER_CHECK(true)
            #define DCHECK_GE(a, b)               LOGGER_CHECK(true)
        #endif

    #endif  // LOGGER_SHUTDOWN end

#else // use glog instead

#undef VLOG_IF
#undef LOG_IF
#undef VLOG
#undef LOG
#undef CHECK
#undef CHECK_NOTNULL
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LT
#undef CHECK_LE
#undef CHECK_GT
#undef CHECK_GE

#undef DVLOG_IF
#undef DLOG_IF
#undef DVLOG
#undef DLOG
#undef DCHECK
#undef DCHECK_NOTNULL
#undef DCHECK_EQ              
#undef DCHECK_NE              
#undef DCHECK_LT              
#undef DCHECK_LE              
#undef DCHECK_GT              
#undef DCHECK_GE              

#endif



#endif // LOGGER_H end
