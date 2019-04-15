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

#ifndef LOGGER_H
#define LOGGER_H

#define LOGGER_SHUTDOWN 0

#include "anakin_config.h"

#ifndef USE_SGX
#include "logger_core.h"

#define SCOPE_LOGGER_CORE_FUNC 		logger::core::funcRegister
#define SCOPE_LOGGER_CORE      		logger::core
#define SCOPE_LOGGER_CORE_CONFIG	logger::core::LoggerConfig

namespace logger {


    inline void init(const char* argv0){
#if not LOGGER_SHUTDOWN
        SCOPE_LOGGER_CORE_FUNC::initial(argv0);
#endif

    }

} // namespace logger


namespace {

#define CHECK_SYMBOL_WARP(name, symbol)    											\
    template <typename T1, typename T2>    											\
    inline std::string * name(const char* expr, const T1& lvalue, const char* op, const T2& rvalue){	\
        if(LOGGER_IS_TRUE(lvalue symbol rvalue)){return nullptr;}					\
        std::ostringstream ss;														\
        ss<<"Check failed:"<< expr <<" ("<<lvalue<<" "<<op<<" "<<rvalue<<") ";		\
        return new std::string(ss.str());											\
    }																				\
    inline std::string * name(const char* expr, int lvalue, const char* op, int rvalue){	\
        return name<int,int>(expr, lvalue, op, rvalue);								\
    }																				\
	inline std::string * name(const char* expr, char lvalue, const char* op, char rvalue){    \
		return name<char,char>(expr, lvalue, op, rvalue);                           \
	}																				\
	inline std::string * name(const char* expr, std::string lvalue, const char* op, std::string rvalue){    \
		return name<std::string,std::string>(expr, lvalue, op, rvalue);             \
	}	
	

CHECK_SYMBOL_WARP(CHECK_EQ_IMPL, ==)
CHECK_SYMBOL_WARP(CHECK_LE_IMPL, <=)
CHECK_SYMBOL_WARP(CHECK_GE_IMPL, >=)
CHECK_SYMBOL_WARP(CHECK_NE_IMPL, !=)
CHECK_SYMBOL_WARP(CHECK_LT_IMPL, <)
CHECK_SYMBOL_WARP(CHECK_GT_IMPL, >)
#undef CHECK_SYMBOL_WARP

/// usage: LOG_S(INFO)<<"function? "<<comevalue<<std::endl;
#define VLOG_IF_S(verbose, cond)																						\
  (SCOPE_LOGGER_CORE::verbose > SCOPE_LOGGER_CORE_CONFIG::current_verbosity_cutoff()									\
			 || (cond)==false) ? (void)0																				\
			 :SCOPE_LOGGER_CORE::voidify() & SCOPE_LOGGER_CORE::loggerMsg(SCOPE_LOGGER_CORE::verbose, __FILE__, __LINE__)
#define LOG_IF_S(verbose_name, cond) VLOG_IF_S(Verbose_##verbose_name, cond)
#define VLOG_S(verbose) VLOG_IF_S(verbose, true)
#define LOG_S(verbose_name) VLOG_S(Verbose_##verbose_name)

/// usage: ABORT_S()<<"error:"<<msg;
#define ABORT_S() SCOPE_LOGGER_CORE::loggerMsg("Abort: ", __FILE__, __LINE__)

#define CHECK_WITH_INFO_S(cond, info)\
    LOGGER_IS_TRUE((cond)==true)?(void)0\
	:SCOPE_LOGGER_CORE::voidify() & SCOPE_LOGGER_CORE::loggerMsg("Check failed: " info " ", __FILE__, __LINE__)
#define CHECK_S(cond) CHECK_WITH_INFO_S(cond, #cond)
#define CHECK_NOTNULL_S(x) CHECK_WITH_INFO_S((x) != nullptr, #x" != nullptr")

#define CHECK_OP_S(func_name, expr1,op,expr2)					 \
    while(auto errStr = func_name(#expr1 " " #op " " #expr2, expr1, #op, expr2)) \
        SCOPE_LOGGER_CORE::loggerMsg(errStr->c_str(), __FILE__, __LINE__)

#define CHECK_EQ_S(A,B) CHECK_OP_S(CHECK_EQ_IMPL,A,==,B)
#define CHECK_NE_S(A,B) CHECK_OP_S(CHECK_NE_IMPL,A,!=,B)
#define CHECK_LE_S(A,B) CHECK_OP_S(CHECK_LE_IMPL,A,<=,B)
#define CHECK_LT_S(A,B) CHECK_OP_S(CHECK_LT_IMPL,A,<,B)
#define CHECK_GE_S(A,B) CHECK_OP_S(CHECK_GE_IMPL,A,>=,B)
#define CHECK_GT_S(A,B) CHECK_OP_S(CHECK_GT_IMPL,A,>,B)

}  // namespace non-name

#if LOGGER_SHUTDOWN
	#undef ENABLE_DEBUG // turn to release mode.
#endif


#ifdef  ENABLE_DEBUG
#define DVLOG_IF_S(verbose, cond)     	VLOG_IF_S(verbose, cond)
#define DLOG_IF_S(verbose_name, cond) 	LOG_IF_S(verbose_name, cond)
#define DVLOG_S(verbose)              	VLOG_S(verbose)
#define DLOG_S(verbose_name)          	LOG_S(verbose_name)
#define DCHECK_S(cond)                  CHECK_S(cond)
#define DCHECK_NOTNULL_S(x)             CHECK_NOTNULL_S(x)
#define DCHECK_EQ_S(a, b)               CHECK_EQ_S(a, b)
#define DCHECK_NE_S(a, b)               CHECK_NE_S(a, b)
#define DCHECK_LT_S(a, b)               CHECK_LT_S(a, b)
#define DCHECK_LE_S(a, b)               CHECK_LE_S(a, b)
#define DCHECK_GT_S(a, b)               CHECK_GT_S(a, b)
#define DCHECK_GE_S(a, b)               CHECK_GE_S(a, b)
#else //BUILD_RELEASE
// log nothing
#define DVLOG_IF_S(verbose, cond)     VLOG_IF_S(verbose, false)
#define DLOG_IF_S(verbose_name, cond) DVLOG_IF_S(Verbose_##verbose_name, cond)
#define DVLOG_S(verbose)              DVLOG_IF_S(verbose,false)
#define DLOG_S(verbose_name)          DVLOG_S(Verbose_##verbose_name)
#define DCHECK_S(cond)                CHECK_S(true || (cond) == true)
#define DCHECK_NOTNULL_S(x)           CHECK_S(true || (x) != nullptr)
#define DCHECK_EQ_S(a, b)             CHECK_S(true || (a) == (b))
#define DCHECK_NE_S(a, b)             CHECK_S(true || (a) != (b))
#define DCHECK_LT_S(a, b)             CHECK_S(true || (a) < (b))
#define DCHECK_LE_S(a, b)             CHECK_S(true || (a) <= (b))
#define DCHECK_GT_S(a, b)             CHECK_S(true || (a) > (b))
#define DCHECK_GE_S(a, b)             CHECK_S(true || (a) >= (b))
#endif


// use local simple logger instead
#ifndef USE_GLOG
#undef LOG
#undef VLOG
#undef LOG_IF
#undef VLOG_IF
#undef CHECK
#undef CHECK_NOTNULL
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LT
#undef CHECK_LE
#undef CHECK_GT
#undef CHECK_GE
#undef DLOG
#undef DVLOG
#undef DLOG_IF
#undef DVLOG_IF
#undef DCHECK
#undef DCHECK_NOTNULL
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DCHECK_LT
#undef DCHECK_LE
#undef DCHECK_GT
#undef DCHECK_GE
#undef VLOG_IS_ON

#if LOGGER_SHUTDOWN // if  LOGGER_SHUTDOWN , turn to release mode.
#define LOG            DLOG_S
#define VLOG           DLOG_S
#define LOG_IF         DLOG_IF_S
#define VLOG_IF        DVLOG_IF_S
#define CHECK(cond)    DCHECK_S((cond))
#define CHECK_NOTNULL  DCHECK_NOTNULL_S
#define CHECK_EQ       DCHECK_EQ_S
#define CHECK_NE       DCHECK_NE_S
#define CHECK_LT       DCHECK_LT_S
#define CHECK_LE       DCHECK_LE_S
#define CHECK_GT       DCHECK_GT_S
#define CHECK_GE       DCHECK_GE_S
#else
#define LOG            LOG_S
#define VLOG           LOG_S
#define LOG_IF         LOG_IF_S
#define VLOG_IF        VLOG_IF_S
#define CHECK(cond)    CHECK_S((cond))
#define CHECK_NOTNULL  CHECK_NOTNULL_S
#define CHECK_EQ       CHECK_EQ_S
#define CHECK_NE       CHECK_NE_S
#define CHECK_LT       CHECK_LT_S
#define CHECK_LE       CHECK_LE_S
#define CHECK_GT       CHECK_GT_S
#define CHECK_GE       CHECK_GE_S
#endif

#define DLOG           DLOG_S
#define DVLOG          DVLOG_S
#define DLOG_IF        DLOG_IF_S
#define DVLOG_IF       DVLOG_IF_S
#define DCHECK         DCHECK_S
#define DCHECK_NOTNULL DCHECK_NOTNULL_S
#define DCHECK_EQ      DCHECK_EQ_S
#define DCHECK_NE      DCHECK_NE_S
#define DCHECK_LT      DCHECK_LT_S
#define DCHECK_LE      DCHECK_LE_S
#define DCHECK_GT      DCHECK_GT_S
#define DCHECK_GE      DCHECK_GE_S
#define VLOG_IS_ON(verbose) ((verbose) <= SCOPE_LOGGER_CORE_CONFIG::current_verbosity_cutoff())
#endif

#else // USE_SGX

// define a nop logger for SGX build
namespace logger {
    inline void init(const char*){}

    struct NopLogger {
        template<typename T>
        constexpr const NopLogger &operator<<(const T &) const {
            return *this;
        }

        template<typename T>
        T *operator&() {
            static_assert(sizeof(T) == 0, "Taking the address of NopLogger is disallowed.");
            return nullptr;
        }
    };

    static constexpr NopLogger __NOP;
}
// namespace logger

#define NOPLOG(X)             logger::__NOP
#define LOG                   NOPLOG
#define VLOG                  NOPLOG
#define DLOG                  NOPLOG
#define CHECK(X)              (((X) == true ? void(nullptr) : abort()), logger::__NOP)
#define CHECK_NOTNULL(X)      CHECK((X) != nullptr)
#define CHECK_EQ(X, Y)        CHECK(((X) == (Y)))
#define CHECK_NE(X, Y)        CHECK(((X) != (Y)))
#define CHECK_LT(X, Y)        CHECK(((X) <  (Y)))
#define CHECK_LE(X, Y)        CHECK(((X) <= (Y)))
#define CHECK_GT(X, Y)        CHECK(((X) >  (Y)))
#define CHECK_GE(X, Y)        CHECK(((X) >= (Y)))
#define ABORT_S()             CHECK(false)

#endif // USE_SGX

#endif // LOGGER_H
