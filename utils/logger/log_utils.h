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
#ifndef LOG_UTILS_H
#define LOG_UTILS_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <cstdarg>
#include <sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/stat.h> // mkdir
#include <unistd.h>   // STDERR_FILENO
#include "anakin_config.h"
#ifdef USE_SGX
#include <support/sgx/sgx_mutex>
#endif

// Disable all warnings from gcc/clang:
#if defined(__clang__)
        #pragma clang system_header
#elif defined(__GNUC__)
        #pragma GCC system_header
#endif

#define LOGGER_CONCAT(str1,str2)  str1##str2

/// \brief intercept the our own abort signal
///
/// for usage: signal(SIGABRT, SIG_DFL);
/// SIG_DFL:default signal handle invoke param
#define LOGGER_CATCH_SIGABRT 1

#define SUPPORT_PTHREADS  1 // support for pthreads

#if defined(ANDROID) || defined(__ANDROID__)
//#ifdef TARGET_ANDROID
	#define STACKTRACES 0
#else
	#define STACKTRACES 1
#endif

#if defined __linux__ || defined __APPLE__
  #include <pthread.h>
  #include <sys/utsname.h>  // For uname.
#ifdef __linux__
#include <linux/limits.h> // PATH_MAX
#endif
#if STACKTRACES
  #include <cxxabi.h>    // for __cxa_demangle for gcc
  #include <dlfcn.h>     // for dladdr
  #include <execinfo.h>  // for backtrace
#endif //STACKTRACES

#if SUPPORT_PTHREADS
  /// @brief
  /// On Linux, the default thread name is the same as the name of the binary.
  /// Additionally, all new threads inherit the name of the thread it got forked from.
  /// For this reason, we use the Thread Local Storage for storing thread names on Linux.
  /// There are many language-specific(c/c++,java,python,perl,objective-c...) and
  /// OS implementions(linux ,window) for TLS.
  #define LOGGER_TLS_NAMES 1
#endif // SUPPORT_PTHREADS

#endif


#ifndef PATH_MAX
        #define PATH_MAX 1024
#endif

#ifdef __APPLE__
        #include "TargetConditionals.h"
#endif

#ifdef  __COUNTER__
  #define LOGGER_ADD_LINE(str)    LOGGER_CONCAT(str,__COUNTER__)
#else
  #define LOGGER_ADD_LINE(str)    LOGGER_CONCAT(str,__LINE__)
#endif
namespace logger {
namespace core {

    enum  VerBoseType: int
    {
        Verbose_OFF     = -9, ///< log off
        Verbose_FATAL   = -3, ///< log info fatal
        Verbose_ERROR   = -2, ///< log info ERROR
        Verbose_WARNING = -1, ///< log info WARNING
        Verbose_INFO    = 0,  ///< log info normal
        Verbose_0       = 0,  ///< verbose  0
        Verbose_1       =1,
        Verbose_2,
        Verbose_3,
        Verbose_4,
        Verbose_5,
        Verbose_6,
        Verbose_7,
        Verbose_8,
        Verbose_9,
        Verbose_Max     =9    ///< max level verbose
    };

    enum FileMode: int {
        CREATE,              ///< CREATE A NEW LOG FILE
        APPEND,              ///< APPEND TO LOCAL FILE(EXISTED!)
    };

    ///< logger start time record
    static const auto startTime = std::chrono::steady_clock::now();

    #if STACKTRACES
    namespace {
        template <class T>
        std::string type_name()
        {
           int status = -1;
           char* demangled = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
           return std::string(status == 0 ? demangled : strdup(typeid(T).name()));
        }
    }
    using PairList = std::vector< std::pair<std::string,std::string> >;
    static const PairList replaceList={
           { type_name<std::string>(),    "std::string"    },
           { type_name<std::wstring>(),   "std::wstring"   },
           { type_name<std::u16string>(), "std::u16string" },
           { type_name<std::u32string>(), "std::u32string" },
           { "std::__1::",                "std::"          },
           { "__thiscall ",               ""               },
           { "__cdecl ",                  ""               },
    };
	#else
	namespace {
        template <class T>
        std::string type_name() { return std::string(""); }
    }
    using PairList = std::vector< std::pair<std::string,std::string> >;
    static const PairList replaceList={};
    #endif

    class ErrContext;
    struct Callback;
    using CallbackVec = std::vector<Callback>;

    namespace  LoggerConfig
    {
		 static bool logtostderr = false;
         static bool colorstderr = true;                    ///< true default
         static bool terminalSupportColor;                  ///< true if terminal support color(win does not support it)
         static unsigned flushBufferInMs = 0;               ///< flush buffer on every line (if x) in x ms later or flush everything immediatly(if 0)
         static bool needFlush = false;
         static std::thread* flushThread = nullptr;         ///< for periodic flushing (a guard thread)
         static std::recursive_mutex  rsMutex;              ///< synchronization primitive used to protect shared data
         static VerBoseType currentVerbos = Verbose_0;
         static VerBoseType currentMaxVerbos = Verbose_OFF;
         static const bool splitFileName = true;            ///< if split the fileName without the path? true ==> pure fileName without path "/"
         static char  projectPath;                          ///< project root path
		 static const char* userName;						///< machine user name , used for log file name.
		 static const char* programName;					///< program name get from argv[0]
		 static const char* hostName;						///< host machine name used for log file name.
	     
         //static pthread_once_t pthreadKeyOnce = PTHREAD_ONCE_INIT;
         #ifdef LOGGER_TLS_NAMES
         static __thread char*  pthreadKeyName;
         #endif

         static __thread  ErrContext* pthreadErrCtPtr;
         const static int threadNameWidth = 16;             ///< Width of the column containing the thread name
         static const int filenNameWidth  = 23;             ///< Width of the column containing the file name

         static std::recursive_mutex  callbackMutex;
         static CallbackVec           callbackVecs;         ///< call back struct vector

	 	 /// return the current max verbos	
	 	 static VerBoseType current_verbosity_cutoff(){
		 	 return currentVerbos> currentMaxVerbos ? 
			 	 currentVerbos : currentMaxVerbos;
	 	 }

         static void init(){

             if (const char* term = getenv("TERM")) {
                if( 0 == strcmp(term, "cygwin")
                            || 0 == strcmp(term, "linux")
                            || 0 == strcmp(term, "screen")
                            || 0 == strcmp(term, "xterm")
                            || 0 == strcmp(term, "xterm-256color")
                            || 0 == strcmp(term, "xterm-color")){
                        terminalSupportColor = true;
                    }
             } else {
                terminalSupportColor = false;
             }

         } // init LoggerConfig
    }// namespace LoggerConfig
} // namespace core


} // namespace logger




#endif // LOG_UTILS_H
