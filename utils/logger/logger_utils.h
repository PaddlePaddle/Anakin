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
#ifndef LOGGER_UTILS_H
#define LOGGER_UTILS_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include <sys/types.h>
#include <sys/stat.h> // mkdir
#include <unistd.h>   // STDERR_FILENO

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

#define SUPPORT_PTHREADS // support for pthreads

#ifndef PLATFORM_ANDROID
  #define ENABLE_STACKTRACES
#endif

#if defined __linux__ || defined __APPLE__
  #include <pthread.h>
  #include <sys/utsname.h>  // For uname.
  #ifdef ENABLE_STACKTRACES
    #include <cxxabi.h>    // for __cxa_demangle for gcc
    #include <dlfcn.h>     // for dladdr
    #include <execinfo.h>  // for backtrace
  #endif //ENABLE_STACKTRACES
#endif

#if defined(_WIN32) || (defined(__APPLE__) && !TARGET_OS_IPHONE) || defined(__linux__)
#   if defined(__APPLE__)
#       define LOGGER_THREAD_LOCAL __thread
#       define LOGGER_TLS_NAMES 1
#   else
#       define LOGGER_THREAD_LOCAL thread_local
#       define LOGGER_TLS_NAMES 1
#   endif
#else
#   error " Anakin can't detect thread local storage."
#   define LOGGER_THREAD_LOCAL
#   define LOGGER_TLS_NAMES 0
#endif

#ifdef  __COUNTER__
  #define LOGGER_ADD_LINE(str)    LOGGER_CONCAT(str,__COUNTER__)
#else
  #define LOGGER_ADD_LINE(str)    LOGGER_CONCAT(str,__LINE__)
#endif

#if defined(__clang__) || defined(__GNUC__)
   /// @brief check the printf var format
   /// @param formatArgs  format args list e.g. 1,2,3,... formatArgs+1
   /// @param firstArg  first format args list, 1 or 2 or ...
   #define LOGGER_CHECK_PRINTF(formatArgs,firstArg)  __attribute__((__format__ (__printf__, formatArgs, firstArg)))
#else
   #define LOGGER_CHECK_PRINTF(formatArgs,firstArg)
#endif

namespace logger {

enum  VerBoseType {
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
    Verbose_Max           ///< max level verbose
};

// out device type
enum DevOutT {
    __OUT,  ///< stdout
    __ERR,  ///< stderr
    __FILE, ///< file
};

// device type
template<DevOutT D>
struct DevType{};

template<>
struct DevType<__OUT> {};

template<>
struct DevType<__ERR> {};

template<>
struct DevType<__FILE> {};

enum ColorEnum {
	RED = 0,
	BLACK,
	BOLD_RED,
	GREEN,
	YELLOW,
	BLUE,
	PURPLE,
	CYAN,
	LIGHT_GRAY,
	WHITE,
	LIGHT_RED,
	DIM,
	BOLD,
	UNDERLINE,
	BLINK,
	RESET	
};

template<ColorEnum C>
struct Color {
	static constexpr char* str = "";
};

template<> struct Color<RED> { static constexpr char* str = "\e[31m"; };
template<> struct Color<BLACK> { static constexpr char* str = "\e[30m"; };
template<> struct Color<BOLD_RED> { static constexpr char* str = "\e[41m"; };
template<> struct Color<GREEN> { static constexpr char* str = "\e[32m"; };
template<> struct Color<YELLOW> { static constexpr char* str = "\e[33m"; };
template<> struct Color<BLUE> { static constexpr char* str = "\e[34m"; };
template<> struct Color<PURPLE> { static constexpr char* str = "\e[35m"; };
template<> struct Color<CYAN> { static constexpr char* str = "\e[36m"; };
template<> struct Color<LIGHT_GRAY> { static constexpr char* str = "\e[37m"; };
template<> struct Color<WHITE> { static constexpr char* str = "\e[37m"; };
template<> struct Color<LIGHT_RED> { static constexpr char* str = "\e[91m"; };
template<> struct Color<DIM> { static constexpr char* str = "\e[2m"; };
template<> struct Color<BOLD> { static constexpr char* str = "\e[1m"; };
template<> struct Color<UNDERLINE> { static constexpr char* str = "\e[4m"; };
template<> struct Color<BLINK> { static constexpr char* str = "\e[5m"; };
template<> struct Color<RESET> { static constexpr char* str = "\e[0m"; };

namespace utils {

/// file i/o mode
struct FileMode {
    enum mode_t {
        CREATE,
        APPEND
    };
    FileMode(mode_t mode):_mode(mode) {}
    const char* c_str() {
        return _mode == CREATE ? "w" : "a";
    }
    operator FileMode::mode_t() {
        return _mode;
    }
    mode_t _mode;
};

template <class T>
std::string type_name() {
#ifdef ENABLE_STACKTRACES
   int status = -1;
   char* demangled = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
   return std::string(status == 0 ? demangled : strdup(typeid(T).name()));
#else
    return "Invalid-Type";
#endif
}

// declare class and class member
class sys {
public:
    static void set_max_logger_verbose_level(VerBoseType verbose);
    static VerBoseType get_max_logger_verbose_level();
    /****************************************************************************/ 
    /*                     system custom signal manipulate                      */ 
    /***************t************************************************************/	
    static void write_to_stderr(const char* data, size_t size);
    static void write_to_stderr(const char* data);
    static void install_logger_signal_handlers();
    
    /****************************************************************************/ 
    /*                     system file manipulate                               */ 
    /***************t************************************************************/	
    static size_t get_file_barrier_size_in_GB();
    static void set_file_barrier_size_in_GB(size_t limit_size_in_GB);
    static void write_file_log(VerBoseType verbose, 
                               const char* expr, 
                               const char* header, 
                               const char* msg);
    static void write_file_log_with_format(VerBoseType verbose, 
                                           const char* expr, 
                                           const char* header, 
                                           const char* format, ...) LOGGER_CHECK_PRINTF(4, 5);
    static void add_file_stream(VerBoseType verbose, void* handle);
    static void file_flush_all();
    static void file_flush_target(VerBoseType verbose);
    static bool logger_to_file(std::string path, FileMode fileMode, VerBoseType verbose);
    static void add_sys_log_file(std::string log_dir);
    
    /****************************************************************************/ 
    /*                     system get basic env info                            */ 
    /***************t************************************************************/	
    static bool colorful_shell_access();
    static void get_username();
    static void get_program_name(const char* argv0);
    static void get_hostname();
    static const char* get_file_name(const char*);
    static char* pick_format(const char* format, va_list vlist);
    static void set_thread_name(const char* name);
    static void get_thread_name(char* buffer, unsigned long long length, bool right_align_hext_id);
    static void init();

    template<VerBoseType Verbose>
    static void add_log_header(char* header, int header_len, const char* file, unsigned line);
    
    
    /****************************************************************************/ 
    /*                     system stack tracing manipulate                      */ 
    /***************t************************************************************/	
    static void transform_stacktrace(std::string& msg);
    static std::string stack_trace(int skip);
};

#define INNER_LOG(Verbose, ...) \
    do {\
        char header[128];\
        sys::add_log_header<Verbose>(header, sizeof(header), __FILE__, __LINE__);\
        ((Verbose) > sys::get_max_logger_verbose_level()) ? \
        (void)0 : sys::write_file_log_with_format(Verbose, "", header, __VA_ARGS__);\
    } while(0)

#define IN_LOG(verbose_name, ...) INNER_LOG(Verbose_##verbose_name, __VA_ARGS__)

class FileStream;
void custom_singal_handler(int signal_number, siginfo_t*, void*);

class SYSResource {
public:
    static SYSResource& Global() {
        static SYSResource _ins;
        return _ins;
    }

    friend class sys;
    friend class FileStream;
    friend void custom_singal_handler(int signal_number, siginfo_t*, void*);

private:
    SYSResource() {
        init_flush_thread();
    }
    ~SYSResource() {}

    void init_flush_thread() {
        std::lock_guard<std::recursive_mutex> lock(this->mut);
        flushFileThread = new std::thread([&, this](){ 
            for (;;) { 
                if (this->needFlushFile) {
                    // if flushFileBufferInMs >0 flush everything!
                    sys::file_flush_all(); 
                } 
                std::this_thread::sleep_for(std::chrono::milliseconds(flushFileBufferInMs)); 
            }
         });
    }

private:
    /// basic info
    bool logtostderr{false};
    bool colorstderr{true};
    /// true if terminal support color(win does not support it)
    bool terminalSupportColor{false};

    /// flush buffer on every line (if x) in x ms later or flush everything immediatly(if 0)
    unsigned flushFileBufferInMs{0};
    bool needFlushFile{true};
    std::thread* flushFileThread{nullptr};
    size_t fileBarrierSizeInGB{10}; // default 1 GPU

    std::string userName;    ///< machine user name , used for log file name.
    std::string programName; ///< program name get from argv[0]
    std::string hostName;    ///< host machine name used for log file name.

    VerBoseType currentMaxVerbos{Verbose_Max};   ///< current max verbose level

public:
    std::recursive_mutex mut; // this can be access by other class or function
};


// namespace holding static resource it looks a little bit trick
namespace SYSStaticRes {
#ifdef LOGGER_TLS_NAMES 
    static LOGGER_THREAD_LOCAL std::string  threadName; ///< customed thread name
#endif
    static const auto startTime = std::chrono::steady_clock::now();
    static const int threadNameWidth = 16;
};

/**
 * \brief  log file stream class
 */
class FileStream {
public:
    static FileStream& Global() {
        static FileStream _ins;
        return _ins;
    }

    void add_file_stream(VerBoseType verbose, void* handle) {
        std::lock_guard<std::recursive_mutex> lock(_file_stream_mut);
        _file_stream_res.push_back(Callback{verbose, handle});
    }

    void write_file_log(VerBoseType verbose, const char* expr, const char* header, const char* msg) {
        std::lock_guard<std::recursive_mutex> lock(_file_stream_mut);
        for(auto& file : _file_stream_res) {
            if(verbose <= file._verbose) {
                FILE* file_p = reinterpret_cast<FILE*>(file._handle);
                int file_desc;
                if((file_desc = fileno(file_p)) == -1) {
                    IN_LOG(FATAL, "fileno error code: %d", file_desc);
                }
                struct stat st; 
                fstat(file_desc, &st); 
                size_t size = st.st_size / (1024*1024*1024);
                size_t barrier_size = sys::get_file_barrier_size_in_GB();
                if(size > barrier_size) {
                    std::fseek(file_p, 0, SEEK_SET); // seek to start
                }
                fprintf(file_p, "%s%s%s\n", header, expr, msg);
                if (SYSResource::Global().flushFileBufferInMs == 0) { 
                    fflush(file_p);
                } else {
                    SYSResource::Global().needFlushFile = true;
                }
            }
        }
    }
   
	~FileStream(){
		close(); 
	}

	void flush_all() {
        std::lock_guard<std::recursive_mutex> lock(_file_stream_mut);
        fflush(stderr);
        for(auto& file : _file_stream_res) {
            FILE* file_p = reinterpret_cast<FILE*>(file._handle); 
            fflush(file_p);
        }
	}

    void flush_target(VerBoseType verbose){
        std::lock_guard<std::recursive_mutex> lock(_file_stream_mut);
        for(auto& file : _file_stream_res) {
            if(file._verbose == verbose) {
                FILE* file_p = reinterpret_cast<FILE*>(file._handle); 
                fflush(file_p);
            }
        }
    }

private:
    FileStream() {}
   
    void close() {
        for(auto& file : _file_stream_res) {
            FILE* file_p = reinterpret_cast<FILE*>(file._handle); 
            fclose(file_p);
        }
	}
    
private:
	struct Callback {
        Callback(VerBoseType v, void* h):_verbose(v), _handle(h){}
        VerBoseType _verbose{Verbose_OFF};
		void* _handle{nullptr}; /// < file handle
	};
    std::vector<Callback> _file_stream_res; /// < file stream resource for log
    std::recursive_mutex  _file_stream_mut;
};

class CustomSignal {
public:
    static CustomSignal& Global() {
        static CustomSignal _ins;
        return _ins;
    }

    void call_default_signal_handler(int signal_number) { 
        struct sigaction sig_action; 
        memset(&sig_action, 0, sizeof(sig_action)); 
        sigemptyset(&sig_action.sa_mask); 
        sig_action.sa_handler = SIG_DFL; // set sig_del for default signal handle 
        sigaction(signal_number, &sig_action, NULL); 
        // send signal to a process . not kill the pthread 
        kill(getpid(), signal_number);
    }

    inline std::string get_custom_name_by_id(int signal_number) {
        for(auto& s : local_singals) {
            if(s.num == signal_number) {
                return s.name;
            }
        } 
        return "";
    }

    void install_custom_signal_handlers() { 
        struct sigaction sig_action; 
        memset(&sig_action, 0, sizeof(sig_action)); 
        sigemptyset(&sig_action.sa_mask); 
        sig_action.sa_flags |= SA_SIGINFO; 
        sig_action.sa_sigaction = &custom_singal_handler; 
        for (const auto& sg : local_singals) { 
            if(sigaction(sg.num, &sig_action, NULL) == -1){ 
                fprintf(stderr,"Failed to install handler for %s\n", sg.name.c_str()); 
            } 
        }
    }

private: 
    CustomSignal() {}
    ~CustomSignal() {}

    struct SignalOwn { 
        int num; 
        std::string name; 
    }; 
    // those signal causes the process to terminate.  
    std::vector<SignalOwn> local_singals { 
#if LOGGER_CATCH_SIGABRT 
        { SIGABRT, "SIGABRT" }, // The SIGABRT signal is sent to a process to tell it to abort( abort().), also can be signal from others.  
#endif 
        { SIGBUS,  "SIGBUS"  }, // incorrect memory access alignment or non-existent physical address 
        { SIGFPE,  "SIGFPE"  }, // (floating-point exception) erroneous arithmetic operation, such as division by zero.  
        { SIGILL,  "SIGILL"  }, // illegal instructionw
        { SIGINT,  "SIGINT"  }, // Terminal interrupt signal.  
        { SIGSEGV, "SIGSEGV" }, // Invalid memory reference.  
        { SIGTERM, "SIGTERM" }, // Termination signal.  
        { SIGPIPE, "SIGPIPE" }, // Write on a pipe with no one to read it.
      };
};

// logger custom signal handler 
inline void custom_singal_handler(int signal_number, siginfo_t*, void*) {
    std::string signal_name("UNKNOWN SIGNAL");
    signal_name = CustomSignal::Global().get_custom_name_by_id(signal_number);

    if(sys::colorful_shell_access()) {
        sys::write_to_stderr(Color<RESET>::str);
        sys::write_to_stderr(Color<BOLD>::str);
        sys::write_to_stderr(Color<CYAN>::str);
    }
    sys::write_to_stderr("\n"); 
    sys::write_to_stderr("logger caught a signal: "); 
    sys::write_to_stderr(signal_name.c_str()); 
    sys::write_to_stderr("\n");
    if(sys::colorful_shell_access()) {
        sys::write_to_stderr(Color<RESET>::str);
    }

    sys::file_flush_all();

    char header[128];
    sys::add_log_header<Verbose_FATAL>(header, sizeof(header), "", 0);
    sys::write_file_log_with_format(Verbose_FATAL, "", header, "Signal caught:", signal_name.c_str());
    auto st = sys::stack_trace(3); 
    if (!st.empty()) {
    	if (sys::colorful_shell_access()) {
        	fprintf(stderr," %s%s%s  >>>>> Fatal error: stack trace: <<<<<< \n %s \n %s\n",
                    Color<RESET>::str,
                    Color<BOLD>::str,
                    Color<GREEN>::str,
                    st.c_str(),
                    Color<RESET>::str);
    	} else {
        	fprintf(stderr," >>>>> Fatal error: stack trace: <<<<<< \n %s \n", st.c_str());
    	}
    	fflush(stderr);
        sys::write_file_log(Verbose_FATAL, "", "", st.c_str());
    }

    CustomSignal::Global().call_default_signal_handler(signal_number);
}

// definition for sys member functions
inline void sys::set_max_logger_verbose_level(VerBoseType verbose = Verbose_0) {
    SYSResource::Global().currentMaxVerbos = verbose;
}
inline VerBoseType sys::get_max_logger_verbose_level() {
    return SYSResource::Global().currentMaxVerbos;
}
/****************************************************************************/ 
/*                     system custom signal manipulate                      */ 
/***************t************************************************************/	
inline void sys::write_to_stderr(const char* data, size_t size) {
    auto result = write(STDERR_FILENO, data, size); 
    (void)result; // Ignore errors.
} 
inline void sys::write_to_stderr(const char* data) {
    sys::write_to_stderr(data, strlen(data));
}
inline void sys::install_logger_signal_handlers() {
    CustomSignal::Global().install_custom_signal_handlers();
}

/****************************************************************************/ 
/*                     system file manipulate                               */ 
/***************t************************************************************/	
inline size_t sys::get_file_barrier_size_in_GB() {
    return SYSResource::Global().fileBarrierSizeInGB;
}
inline void sys::set_file_barrier_size_in_GB(size_t limit_size_in_GB) {
    SYSResource::Global().fileBarrierSizeInGB = limit_size_in_GB;
}
inline void sys::write_file_log(VerBoseType verbose, 
                                const char* expr, 
                                const char* header, 
                                const char* msg) {
    FileStream::Global().write_file_log(verbose, expr, header, msg);
}
inline void sys::write_file_log_with_format(VerBoseType verbose, 
                                            const char* expr, 
                                            const char* header, 
                                            const char* format, ...) {
    va_list vlist; 
    va_start(vlist, format); 
    auto msg = sys::pick_format(format, vlist);
    FileStream::Global().write_file_log(verbose, expr, header, msg);
    free(msg);
    msg = nullptr;
    va_end(vlist);
}
inline void sys::add_file_stream(VerBoseType verbose, void* handle) {
    FileStream::Global().add_file_stream(verbose, handle);
}
inline void sys::file_flush_all() {
    FileStream::Global().flush_all();
}
inline void sys::file_flush_target(VerBoseType verbose){
    FileStream::Global().flush_target(verbose);
} 
inline bool sys::logger_to_file(std::string path, FileMode fileMode, VerBoseType verbose) { 
    char* file_path = strdup(path.c_str()); 
    for (char* p = strchr(file_path + 1, '/'); p!=NULL; p = strchr(p + 1, '/')){ 
        *p = '\0'; 
        struct stat st; 
        if((stat(file_path, &st) == 0) && (((st.st_mode) & S_IFMT) == S_IFDIR)){ 
            // file_path exists and is a directory. do nothing 
            *p = '/'; 
            continue; 
        } 
        else { 
            if(mkdir(file_path,0755)==-1){ 
                IN_LOG(ERROR,"Failed to ceate the path '%s'\n", file_path); 
                return false; 
            } 
        } 
        *p = '/'; 
    } 
    free(file_path); 
    auto file = fopen(path.c_str(), fileMode.c_str()); 
    if(!file){ 
        IN_LOG(ERROR, "Failed to open '%s'\n", path.c_str()); 
        return false; 
    } 
    // register to file stream
    sys::add_file_stream(verbose, file);
    if (fileMode == FileMode::APPEND) { 
        fprintf(file,"\n\n\n\n\n"); 
        fflush(file); 
    } 
    fprintf(file, "File log level: %d\n", verbose); 
    fprintf(file, " V  |     time     |  uptime  | %s | %s:line ] \n", " thread name/id", "File"); 
    fflush(file); 
    IN_LOG(INFO, "Logging to '%s', FileMode: '%s', Level: %d\n", path.c_str(), fileMode.c_str(), verbose); 
    fflush(stderr);
    return true; 
}
/// @brief same as glog.
/// If no base filename for logs of this severity has been set, use a default base filename of
/// "<program name>.<hostname>.<user name>.log.<severity level>.".  So
/// logfiles will have names like
/// webserver.examplehost.root.log.INFO.19990817-150000.4354 or ....thread_name, where
/// 19990817 is a date (1999 August 17), 150000 is a time (15:00:00),
/// and 4354 is the pid of the logging process or thread_name of the logging thread.
/// The date & time reflect when the file was created for output.
/// Where does the file get put?  Successively try the directories
/// "/tmp", and "." , "/tmp" is default path for logging.
/// add_sys_log_file will be invoked in the end of initial function.
inline void sys::add_sys_log_file(std::string log_dir = "/tmp"){
    // default filename
    std::string filename = ""; 
    filename += SYSResource::Global().programName + "."; 
    filename += SYSResource::Global().hostName + "."; 
    filename += SYSResource::Global().userName + "."; 
    filename += "log.";
    filename = log_dir + "/" + filename; 
    long long ms_since_epoch = \
                               std::chrono::duration_cast<std::chrono::milliseconds>(\
                                       std::chrono::system_clock::now().time_since_epoch()).count(); 
    time_t sec_since_epoch = time_t(ms_since_epoch / 1000); 
    tm time_info; 
    localtime_r(&sec_since_epoch, &time_info); 
                        
    char thread_name[SYSStaticRes::threadNameWidth + 1] = {0}; 
    get_thread_name(thread_name, SYSStaticRes::threadNameWidth + 1, true); 
                                
    char buff[20 + SYSStaticRes::threadNameWidth]; 
    int buff_size = sizeof(buff)/sizeof(char); 
    snprintf(buff, buff_size, ".%04d%02d%02d-%02d%02d%02d.%s", 
                              1900 + time_info.tm_year, 1 + time_info.tm_mon, time_info.tm_mday, 
                              time_info.tm_hour, time_info.tm_min, time_info.tm_sec, thread_name);

    auto get_log_filename = [&](std::string verbose_str) -> std::string { 
        //printf("path :: %s \n ",(filename + verbose_str + std::string(buff)).c_str());  // for debug 
        return std::string(filename + verbose_str) + std::string(buff);
    };

    // create the log file with diff verbose 
    sys::logger_to_file(get_log_filename("FATAL"),FileMode::CREATE,VerBoseType::Verbose_FATAL); 
    sys::logger_to_file(get_log_filename("ERROR"),FileMode::CREATE,VerBoseType::Verbose_ERROR); 
    sys::logger_to_file(get_log_filename("WARNING"),FileMode::CREATE,VerBoseType::Verbose_WARNING); 
    sys::logger_to_file(get_log_filename("INFO"),FileMode::CREATE,VerBoseType::Verbose_INFO);
}

/****************************************************************************/ 
/*                     system get basic env info                            */ 
/***************t************************************************************/	
inline bool sys::colorful_shell_access() {
    return SYSResource::Global().colorstderr \
        && SYSResource::Global().terminalSupportColor;
}
inline void sys::get_username() { 
    const char* user = getenv("USER"); 
    if (user != NULL) { 
        SYSResource::Global().userName = user; 
    } else { 
        SYSResource::Global().userName = "invalid-user"; 
    } 
}
inline void sys::get_program_name(const char* argv0) { 
    const char* slash = strrchr(argv0, '/');
    SYSResource::Global().programName = slash ? slash+1 : argv0; 
}
inline void sys::get_hostname() { 
#if defined __linux__ || defined __APPLE__ 
    struct utsname buf; 
    if(0 != uname(&buf)){ 
        // ensure null termination on failure 
        *buf.nodename = '\0'; 
    } 
    SYSResource::Global().hostName = strdup(buf.nodename);
#else
    # warning There is no way to retrieve the host name(not support os windows).
    SYSResource::Global().hostName = "(unknown)";
#endif  
}
inline char* sys::pick_format(const char* format, va_list vlist) { 
	char* msg = nullptr;
	int result = vasprintf(&msg, format, vlist);
	if(result == -1){
		IN_LOG(ERROR, "Bad string format: '%s'\n", format);
	}
	return msg; 
}
inline void sys::set_thread_name(const char* name) { 
#ifdef LOGGER_TLS_NAMES 
    SYSStaticRes::threadName = std::string(name); 
#else 
#ifdef __APPLE__ 
    pthread_setname_np(name); 
#else 
    pthread_setname_np(pthread_self(), name); 
#endif 
#endif
}
inline void sys::get_thread_name(char* buffer, unsigned long long length, bool right_align_hext_id = true) {
    if(length == 0u){
	    IN_LOG(ERROR, "get_thread_name get 0 length buffer. ");
	}
	if(buffer == nullptr){
	    IN_LOG(ERROR, "get_thread_name get nullptr buffer. ");
	} 
#ifdef SUPPORT_PTHREADS 
    auto thread = pthread_self(); 
#ifdef LOGGER_TLS_NAMES 
    if (const char* name = SYSStaticRes::threadName.c_str()) { 
        snprintf(buffer, length, "%s", name); 
    } else { 
        buffer[0] = 0; 
    } 
#else 
    pthread_getname_np(thread, buffer, length); 
#endif 
    if (buffer[0] == 0) { 
#ifdef __APPLE__ 
        uint64_t thread_id; 
        pthread_threadid_np(thread, &thread_id); 
#else 
        uint64_t thread_id = thread; 
#endif 
        if (right_align_hext_id) { 
            snprintf(buffer, length, "%*X", (unsigned)length - 1, static_cast<unsigned>(thread_id)); 
        } else { 
            snprintf(buffer, length, "%X", static_cast<unsigned>(thread_id)); 
        } 
    } 
#else // SUPPORT_PTHREADS 
    buffer[0] = 0; 
#endif // SUPPORT_PTHREADS
}
inline const char* sys::get_file_name(const char* path) { 
    for (auto ptr = path; *ptr; ++ptr) { 
        if (*ptr == '/' || *ptr == '\\') { 
            path = ptr + 1; 
        } 
    } 
    return path;
}
inline void sys::init() {
    if(const char* term = getenv("TERM")) {
        if( 0 == strcmp(term, "cygwin") || 0 == strcmp(term, "linux")
                                        || 0 == strcmp(term, "screen")
                                        || 0 == strcmp(term, "xterm")
                                        || 0 == strcmp(term, "xterm-256color")
                                        || 0 == strcmp(term, "xterm-color")) {
                    SYSResource::Global().terminalSupportColor = true; 
        } else {
            SYSResource::Global().terminalSupportColor = false;
        }
    }
}
template<VerBoseType Verbose>
inline void sys::add_log_header(char* header, int header_len, const char* file, unsigned line) { 
    long long ms_since_epoch = \
                  std::chrono::duration_cast<std::chrono::milliseconds>(\
                          std::chrono::system_clock::now().time_since_epoch()).count();
    time_t sec_since_epoch = time_t(ms_since_epoch / 1000);
    tm time_info;
    localtime_r(&sec_since_epoch, &time_info);

    auto uptime_ms = \
                  std::chrono::duration_cast<std::chrono::milliseconds>(\
                          std::chrono::steady_clock::now() - SYSStaticRes::startTime).count();
    auto uptime_sec = uptime_ms / 1000.0;

    char thread_name[SYSStaticRes::threadNameWidth + 1] = {0};
    sys::get_thread_name(thread_name, SYSStaticRes::threadNameWidth + 1, true);

    file = sys::get_file_name(file);

    char level_buff[6];
    if (Verbose <= Verbose_FATAL) { 
        snprintf(level_buff, sizeof(level_buff) - 1, "Ftl");
    } else if (Verbose == Verbose_ERROR) { 
        snprintf(level_buff, sizeof(level_buff) - 1, "Err"); 
    } else if (Verbose == Verbose_WARNING) { 
        snprintf(level_buff, sizeof(level_buff) - 1, "War"); 
    } else if (Verbose == Verbose_INFO) {
        snprintf(level_buff, sizeof(level_buff) - 1, "Inf");
    } else { 
        snprintf(level_buff, sizeof(level_buff) - 1, "%3d", Verbose);
    }
    // fill the header
    snprintf(header, header_len, "%4s| %02d:%02d:%02d.%05lld| %.3fs| %s| %s:%u] ",
             level_buff,
             time_info.tm_hour, time_info.tm_min, time_info.tm_sec, ms_since_epoch % 1000,
             uptime_sec,
             thread_name,
             file,
             line); 
}

/****************************************************************************/ 
/*                     system stack tracing manipulate                      */ 
/***************t************************************************************/	
inline void sys::transform_stacktrace(std::string& msg) {
#ifdef ENABLE_STACKTRACES
    std::vector< std::pair<std::string,std::string> > patterns={
       { type_name<std::string>(),    "std::string"    },
       { type_name<std::wstring>(),   "std::wstring"   },
       { type_name<std::u16string>(), "std::u16string" },
       { type_name<std::u32string>(), "std::u32string" },
       { "std::__1::",                "std::"          },
       { "__thiscall ",               ""               },
       { "__cdecl ",                  ""               },
    };
    for(auto& pattern : patterns) {
        if(pattern.first.size() > pattern.second.size()) {
            size_t pos; 
            while((pos=msg.find(pattern.first)) != std::string::npos) {
                msg.replace(pos, pattern.first.size(), pattern.second);
            }
        }
    }
    try {
        std::regex std_allocator_re(R"(,\s*std::allocator<[^<>]+>)");
        msg = std::regex_replace(msg, std_allocator_re, std::string(""));
        std::regex template_spaces_re(R"(<\s*([^<> ]+)\s*>)");
        msg = std::regex_replace(msg, template_spaces_re, std::string("<$1>"));
    } catch (std::regex_error&) {/*may throw exception*/}
#endif
}
/// we use libstdc++'s abi::__cxa_demangle() to set friendly demangled stack trace.
/// Example:
///
///     | Mangled Name  | abi::__cxa_demangle()
///     |---------------|-----------------------
///     | _Z1fv         | f()
///     | _Z1fi         | f(int)
///     | _Z3foo3bar    | foo(bar)
///     | _Z1fIiEvi     | void f<int>(int)
///     | _ZN1N1fE      | N::f
///     | _ZN3Foo3BarEv | Foo::Bar()
///     | _Zrm1XS_"     | operator%(X, X)
///     | _ZN3FooC1Ev   | Foo::Foo()
///     | _Z1fSs        | f(std::basic_string<char,
///     |               |   std::char_traits<char>,
///     |               |   std::allocator<char> >)
inline std::string sys::stack_trace(int skip) { 
#ifdef ENABLE_STACKTRACES
    void* callstack[256];  // max trace deep [256]
    const auto max_frames = sizeof(callstack) / sizeof(callstack[0]); 
    int num_frames = backtrace(callstack, max_frames); 
    char** symbols = backtrace_symbols(callstack, num_frames);
    std::string result;
    // reverse the stack trace result
    for (int i = num_frames - 1; i >= skip; i--) {
        char buf[1024]; // frame buffer
        // typedef struct {
        //          const char *dli_fname;  /* Pathname of shared object that contains address */
        //          void       *dli_fbase;  /* Address at which shared object is loaded */
        //          const char *dli_sname;  /* Name of nearest symbol with address lower than addr */
        //          void       *dli_saddr;  /* Exact address of symbol named in dli_sname */
        // } Dl_info;
        // If no symbol matching addr could be found, then dli_sname and dli_saddr are set to NULL.
        // The function dladdr() takes a function pointer and tries to resolve name and file where it is located
        Dl_info info;
        if (dladdr(callstack[i], &info) && info.dli_sname) { 
            char* demangled = NULL; 
            int status = -1; 
            if (info.dli_sname[0] == '_') { 
                demangled = abi::__cxa_demangle(info.dli_sname, 0, 0, &status); 
            } 
            snprintf(buf, sizeof(buf), " %-3d %*p -> %s + %zd\n", 
                                       i - skip, int(2 + sizeof(void*) * 2), callstack[i], 
                                       status == 0 ? demangled : 
                                       info.dli_sname == 0 ? symbols[i] : info.dli_sname, 
                                       static_cast<char*>(callstack[i]) - static_cast<char*>(info.dli_saddr)); 
            free(demangled); 
        } else { 
            snprintf(buf, sizeof(buf), "%-3d %*p -> %s\n",
                                       i - skip, int(2 + sizeof(void*) * 2), callstack[i], 
                                       symbols[i]); 
        } 
        result += buf;
    } // for
    free(symbols);
    if (num_frames == max_frames) {
        result = "[logger truncated]\n" + result;
    }
    if (!result.empty() && result[result.size() - 1] == '\n') {
        result.resize(result.size() - 1);
    }
    sys::transform_stacktrace(result);
    return result;
#else
    return "";
#endif
}

} /* namespace utils */

} /* namespace logger */

#endif
