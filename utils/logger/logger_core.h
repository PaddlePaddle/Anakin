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

#ifndef LOGGER_CORE_H
#define LOGGER_CORE_H
#include "log_utils.h"

namespace logger {

 namespace  core{

   using namespace std::chrono;
#if defined(__clang__) || defined(__GNUC__)
   /// @brief check the printf var format
   /// @param formatArgs  format args list e.g. 1,2,3,... formatArgs+1
   /// @param firstArg  first format args list, 1 or 2 or ...
   #define LOGGER_CHECK_PRINTF(formatArgs,firstArg)  __attribute__((__format__ (__printf__, formatArgs, firstArg)))
#else
   #define LOGGER_CHECK_PRINTF(formatArgs,firstArg)
#endif
   //#define LOGGER_NORET __attribute__((noreturn))
   #define LOGGER_NORET


   /// judge x if false or true
   #define LOGGER_IS_FALSE(x) (__builtin_expect(x,0))
   #define LOGGER_IS_TRUE(x)  (__builtin_expect(!!(x),1))

   /// __PRETTY_FUNCTION__ (c++)return the type signature of the function as well as its bare name.
   #define LOGGER_PRETTY_FUNC  __PRETTY_FUNCTION__

      /**
       *  @brief  text class: hold char* type
       *
       */
      class text
      {
      public:
        explicit text(char* str):str_(str){}
        ~text(){free(str_);}

        text(text&& another)
        {
          str_ = another.str_;
          another.str_ = nullptr;
        }

        text(text&) = delete;
        text& operator =(text& a) = delete;

        const char* c_str() const {return str_;}
        bool empty() const { return str_== nullptr || *str_ == '\0';}

        char* pop()
        {
          auto result = str_;
          str_ = nullptr;
          return result;
        }
      private:
        char* str_;
      };
      /**
       *  @brief  basic message struct
       *
       */
      struct Message
      {
        VerBoseType   verbose;   // Already part of preamble
        const char* filename;    // Already part of preamble
        unsigned    line;        // Already part of preamble
        const char* preamble;    // Date, time, uptime, thread, file:line, verbose.
        const char* prefix;      // Assertion failure info goes here (or "").
        const char* message;     // User message goes here.
      };

      /**
       *  @brief  basic message struct
       *
       */
      class ErrContext
      {
      public:
        ErrContext(const char* file, unsigned line, const char* descr)
                  :file_(file), line_(line), descr_(descr){
          ErrContext*& head = LoggerConfig::pthreadErrCtPtr;
          previous_ = head;
          head = this;   ///< add head in construction
        }
        ~ErrContext(){
          LoggerConfig::pthreadErrCtPtr = previous_;    ///< remove head
        }
        /// desable assign and copy (lvalue && rvalue)
        ErrContext( const ErrContext&) = delete;
        ErrContext(ErrContext&&) = delete;
        ErrContext& operator=(const ErrContext&) = delete;
        ErrContext& operator=(ErrContext&&) = delete;

        ErrContext* previous() const {return previous_;}
      public:
        const char* file_;
        unsigned    line_;
        const char* descr_;
        ErrContext* previous_;
      };
      /// used for callback
      typedef void (*log_handler_t)(void* user_data, const Message& message);
      typedef void (*close_handler_t)(void* user_data);
      typedef void (*flush_handler_t)(void* user_data);

      /**
       *  @brief  basic callback struct
       *
       */
      class Callback
      {
	  public:
		~Callback(){
			close_handler_t(file);
		}
        std::string     id;
        log_handler_t   callback;
        void*           user_data;
        VerBoseType     verbose; // not change!
        close_handler_t close;
        flush_handler_t flush;
      };
      /// \brief the signal to set the signal handler to.
      ///
      /// It can be an implementation-defined value or one of the following values:
      /// SIGABRT
      /// SIGFPE
      /// SIGILL
      /// SIGINT
      /// SIGSEGV
      /// SIGTERM
      struct SignalOwn {
        int sigNum;
        const char* sigName;
      };

	  // those signal causes the process to terminate.
      static const SignalOwn LOCAL_SIGS[] = {
      #if LOGGER_CATCH_SIGABRT
                  { SIGABRT, "SIGABRT" }, // The SIGABRT signal is sent to a process to tell it to abort( abort().), also can be signal from others.
      #endif
                  { SIGBUS,  "SIGBUS"  }, // incorrect memory access alignment or non-existent physical address
                  { SIGFPE,  "SIGFPE"  }, // (floating-point exception) erroneous arithmetic operation, such as division by zero.
                  { SIGILL,  "SIGILL"  }, // illegal instruction
                  { SIGINT,  "SIGINT"  }, // Terminal interrupt signal.
                  { SIGSEGV, "SIGSEGV" }, // Invalid memory reference.
                  { SIGTERM, "SIGTERM" }, // Termination signal.
				  { SIGPIPE, "SIGPIPE" }, // Write on a pipe with no one to read it.
      };

      /**
       *  @brief  func register
       *
       */
      namespace funcRegister
      {
        /****************************************************************************/
        /*                            logger init                                   */
        /***************t************************************************************/
        void initial(const char* argv0);
        /****************************************************************************/
        /*                            function register                             */
        /***************t************************************************************/
        inline const char* black();
        inline const char* red();
        inline const char* green();
        inline const char* yellow();
        inline const char* blue();
        inline const char* purple();
        inline const char* cyan();
        inline const char* light_gray();
        inline const char* white();
        inline const char* light_red();
        inline const char* dim();
        // msg format
        inline const char* bold();
        inline const char* underline();
        inline const char* blink();
        // colorful terminal should end whit reset!
        inline const char* reset();

        /****************************************************************************/
        /*                            log file manipulate                           */
        /***************t************************************************************/
		void get_username();
		void get_program_name(const char* argv0);
		void get_hostname();
        void flush_callback();
        void on_callback_change();
        bool remove_callback(const char* id);
        void add_callback(const char* id,
                          log_handler_t callback,
                          void* user_data,
                          VerBoseType verbose,
                          close_handler_t on_close,
                          flush_handler_t on_flush);
		void add_sys_log_file(std::string log_dir);
        bool logger_to_file(const char* path,FileMode fileMode,VerBoseType verbose);
        void file_log(void* user_data, const Message& message);
        void file_close(void* user_data);
        void file_flush(void* user_data);
        const char* filename(const char* path);

        /****************************************************************************/
        /*                            error context manipulate                      */
        /***************t************************************************************/
        text get_error_context();
        text get_error_context_for(const ErrContext* head);
        /****************************************************************************/
        /*                            stack tracing manipulate                      */
        /***************t************************************************************/
        void do_replacements(const PairList& replacements, std::string& str);
        std::string set_friendly_stacktrace(const std::string& input);
        std::string stacktrace_as_stdstring(int skip);
        text stacktrace(int skip);
        /****************************************************************************/
        /*                            multi pthread manipulate                      */
        /***************t************************************************************/
        void set_thread_name(const char* name);
        void get_thread_name(char* buffer, unsigned long long length, bool right_align_hext_id);
        /****************************************************************************/
        /*                            log stderr manipulate                         */
        /****************************************************************************/
        void print_preamble(char* out_buff, size_t out_buff_size, VerBoseType    verbose, const char* file, unsigned line);
        void log_message(int stack_trace_skip, Message& message, bool abort_if_fatal);
        void log_to_all(int            stack_trace_skip,
                               VerBoseType    verbose,
                               const char*    file,
                               unsigned       line,
                               const char*    prefix,
                               const char*    buff);
        text vtextprintf(const char* format, va_list vlist);
        /// attribute fmtArg check shuld +1 when used in class(skip the "this" point)
        /// but for class , static members is the same as non-members,record as below:
        /// + non-member functions work with 1,2
        /// + static member functions work with 1,2
        /// + non-static member functions treat 'this' as #1, so need 2,3
        text textprintf(const char* format, ...) LOGGER_CHECK_PRINTF(1, 2);
        void log(VerBoseType verbose, const char* file, unsigned line, const char* format, ...) LOGGER_CHECK_PRINTF(4, 5);
        LOGGER_NORET void log_and_abort(int stack_trace_skip, const char* expr, const char* file, unsigned line, const char* format, ...) LOGGER_CHECK_PRINTF(5, 6);
        /****************************************************************************/
        /*                             signal manipulate                            */
        /****************************************************************************/
        void write_to_stderr(const char* data, size_t size);
        void write_to_stderr(const char* data);
        void call_default_signal_handler(int signal_number);
        void logger_signal_handler(int signal_number, siginfo_t*, void*);
        void install_logger_signal_handlers();
      }
      /**
       *  @brief  logger class
       *
       */
      class loggerMsg
      {
      public:
        loggerMsg(VerBoseType verbose, const char* file, unsigned line)
                  :verbose_(verbose),file_(file),line_(line),isAbort_(false){}

        loggerMsg(const char* expression, const char* file, unsigned line)
                  :expression_(expression),file_(file),line_(line),isAbort_(true){}
        ~loggerMsg();

        template<typename T>
        loggerMsg& operator<<(const T& var){
          ss_<<var;
          return *this;
        }

        // access for std::endl and other io
        loggerMsg& operator<<(std::ostream&(*func)(std::ostream&)){
          func(ss_);
          return *this;
        }
      private:
        bool isAbort_;
        VerBoseType verbose_;
        const char* expression_;
        const char* file_;
        unsigned    line_;
        std::ostringstream ss_;
      };
	  /**
       * @brief voidify the class such as logger for macro defines
       *
       * usage: voidify()(loggerMsg(...)).
       */ 
      class voidify
      {
      public:
          voidify(){}
          void operator&(const loggerMsg&){}
      };

      using CallbackVec = std::vector<Callback>;
      using StrPair     = std::pair<std::string, std::string>;
      using StrPairList = std::vector<StrPair>;
	   
      #define INNER_LOG(verbose,...)										   \
		((verbose) > LoggerConfig::current_verbosity_cutoff()) ? (void)0       \
                : funcRegister::log(verbose, __FILE__, __LINE__, __VA_ARGS__)
      #define IN_LOG(verbose_name, ...) INNER_LOG(Verbose_##verbose_name, __VA_ARGS__)

  } // namespace core

}  // namespace logger

namespace logger {

namespace core {

namespace funcRegister {

inline const char* black()      { return LoggerConfig::terminalSupportColor ? "\e[30m" : ""; }
inline const char* red()        { return LoggerConfig::terminalSupportColor ? "\e[31m" : ""; }
inline const char* b_red()	{ return LoggerConfig::terminalSupportColor ? "\e[41m" : ""; }
inline const char* green()      { return LoggerConfig::terminalSupportColor ? "\e[32m" : ""; }
inline const char* yellow()     { return LoggerConfig::terminalSupportColor ? "\e[33m" : ""; }
inline const char* blue()       { return LoggerConfig::terminalSupportColor ? "\e[34m" : ""; }
inline const char* purple()     { return LoggerConfig::terminalSupportColor ? "\e[35m" : ""; }
inline const char* cyan()       { return LoggerConfig::terminalSupportColor ? "\e[36m" : ""; }
inline const char* light_gray() { return LoggerConfig::terminalSupportColor ? "\e[37m" : ""; }
inline const char* white()      { return LoggerConfig::terminalSupportColor ? "\e[37m" : ""; }
inline const char* light_red()  { return LoggerConfig::terminalSupportColor ? "\e[91m" : ""; }
inline const char* dim()        { return LoggerConfig::terminalSupportColor ? "\e[2m"  : ""; }
inline const char* bold()       { return LoggerConfig::terminalSupportColor ? "\e[1m" : ""; }
inline const char* underline()  { return LoggerConfig::terminalSupportColor ? "\e[4m" : ""; }
inline const char* blink()      { return LoggerConfig::terminalSupportColor ? "\e[5m" : "";}
inline const char* reset()      { return LoggerConfig::terminalSupportColor ? "\e[0m" : ""; }

inline void get_username() {
  const char* user = getenv("USER");
  if (user != NULL) {
    LoggerConfig::userName = user;
  } else {
    LoggerConfig::userName = "invalid-user";
  }
}

inline void get_program_name(const char* argv0) {
  const char* slash = strrchr(argv0, '/');
  LoggerConfig::programName = slash ? slash+1 : argv0;
}

inline void get_hostname() {
#if defined __linux__ || defined __APPLE__
  struct utsname buf;
  if(0 != uname(&buf)){
	// ensure null termination on failure
	*buf.nodename = '\0';
  }
  LoggerConfig::hostName = strdup(buf.nodename);
#else
# warning There is no way to retrieve the host name(not support os windows).
  LoggerConfig::hostName = "(unknown)";
#endif
}

// @brief same as glog...
// If no base filename for logs of this severity has been set, use a default base filename of
// "<program name>.<hostname>.<user name>.log.<severity level>.".  So
// logfiles will have names like
// webserver.examplehost.root.log.INFO.19990817-150000.4354 or ....thread_name, where
// 19990817 is a date (1999 August 17), 150000 is a time (15:00:00),
// and 4354 is the pid of the logging process or thread_name of the logging thread.  
// The date & time reflect when the file was created for output.
// Where does the file get put?  Successively try the directories
// "/tmp", and "." , "/tmp" is default path for logging.
// add_sys_log_file will be invoked in the end of initial function.  
inline void add_sys_log_file(std::string log_dir = "/tmp"){
	// default filename
	std::string filename = ""; 
	filename += std::string(LoggerConfig::programName) + ".";	
	filename += std::string(LoggerConfig::hostName) + ".";
	filename += std::string(LoggerConfig::userName) + ".";
	filename += "log.";
	// now we get <program name>.<hostname>.<user name>.log.
	filename = log_dir + "/" + filename;

	long long ms_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  	time_t sec_since_epoch = time_t(ms_since_epoch / 1000);
  	tm time_info;
  	localtime_r(&sec_since_epoch, &time_info);
	
	char thread_name[LoggerConfig::threadNameWidth + 1] = {0};
    get_thread_name(thread_name, LoggerConfig::threadNameWidth + 1, true);

	char buff[20+LoggerConfig::threadNameWidth];
	int buff_size = sizeof(buff)/sizeof(char);
	snprintf(buff, buff_size, ".%04d%02d%02d-%02d%02d%02d.%s",
           1900 + time_info.tm_year, 1 + time_info.tm_mon, time_info.tm_mday,
           time_info.tm_hour, time_info.tm_min, time_info.tm_sec, thread_name);
	
	auto get_log_filename = [&](std::string verbose_str) -> const char* {
		//printf("path :: %s \n ",(filename + verbose_str + std::string(buff)).c_str());  // for debug
		return strdup((filename + verbose_str + std::string(buff)).c_str());
	};

	// create the log file with diff verbose
	logger_to_file(get_log_filename("FATAL"),FileMode::CREATE,VerBoseType::Verbose_FATAL);
	logger_to_file(get_log_filename("ERROR"),FileMode::CREATE,VerBoseType::Verbose_ERROR);
	logger_to_file(get_log_filename("WARNING"),FileMode::CREATE,VerBoseType::Verbose_WARNING);
	logger_to_file(get_log_filename("INFO"),FileMode::CREATE,VerBoseType::Verbose_INFO);
	//logger_to_file(get_log_filename("Verbose_0"),FileMode::CREATE,VerBoseType::Verbose_0);
}

inline void flush_callback(){
  std::lock_guard<std::recursive_mutex> lock(LoggerConfig::callbackMutex);
  fflush(stderr); // fflush to terminal first
  for (const auto& callback : LoggerConfig::callbackVecs)
  {
          if (callback.flush) {
                  callback.flush(callback.user_data);
          }
  }
  LoggerConfig::needFlush = false;

}

inline void on_callback_change(){
  LoggerConfig::currentMaxVerbos = Verbose_OFF; // min verbos value
  for (const auto& callback : LoggerConfig::callbackVecs)
  {
    LoggerConfig::currentMaxVerbos = std::max(LoggerConfig::currentMaxVerbos, callback.verbose);
  }
}

inline bool remove_callback(const char* id){
  std::lock_guard<std::recursive_mutex> lock(LoggerConfig::callbackMutex);
  auto it = std::find_if(begin(LoggerConfig::callbackVecs), end(LoggerConfig::callbackVecs), [&](const Callback& c) { return c.id == id; });
  if (it != LoggerConfig::callbackVecs.end()) {
    if (it->close) { it->close(it->user_data); } // close file ptr
    LoggerConfig::callbackVecs.erase(it);
    on_callback_change();
    return true;
  } else {
	IN_LOG(ERROR,"Failed to locate callback with id '%s'\n", id);
    return false;
  }
}

inline void add_callback(const char* id,
                  log_handler_t callback,
                  void* user_data,
                  VerBoseType verbose,
                  close_handler_t on_close,
                  flush_handler_t on_flush)
{
  std::lock_guard<std::recursive_mutex> lock(LoggerConfig::callbackMutex);
  LoggerConfig::callbackVecs.push_back(Callback{id, callback, user_data, verbose, on_close, on_flush});
  on_callback_change();
}

inline bool logger_to_file(const char* path,FileMode fileMode,VerBoseType verbose)
{
  char* file_path = strdup(path);
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
             IN_LOG(ERROR,"Failed to ceate the path '%s'\n",file_path);
             return false;
        }
	  }
      *p = '/';
  }
  free(file_path);
  const char* modeStr = (fileMode == FileMode::CREATE ? "w" : "a");
  auto file = fopen(path, modeStr);
  if(!file){
      IN_LOG(ERROR,"Failed to open '%s'\n",path);
      return false;
  }
  // add new callback int to vector static
  add_callback(path, file_log, file, verbose, file_close, file_flush);
  if (fileMode == FileMode::APPEND) {
      fprintf(file,"\n\n\n\n\n");
      fflush(file);
  }
  fprintf(file, "File verbosity level: %d\n", verbose);
  auto PREAMBLE_EXPLAIN = textprintf("  v  |     time     |  uptime  | %s | %s:line ] ", " thread name/id", "file");
  fprintf(file, "%s\n", PREAMBLE_EXPLAIN.c_str());
  fflush(file);
  IN_LOG(INFO,"Logging to '%s', FileMode: '%s', VerBoseType: %d\n",path, modeStr, verbose);
  fflush(stderr);

  return true;
}

inline void file_log(void* user_data, const Message& message)
{
  FILE* file = reinterpret_cast<FILE*>(user_data);
  fprintf(file, "%s%s%s\n",message.preamble, message.prefix, message.message);
  if (LoggerConfig::flushBufferInMs) {
    fflush(file);
  }
}

inline void file_close(void* user_data)
{
  FILE* file = reinterpret_cast<FILE*>(user_data);
  fclose(file);
}

inline void file_flush(void* user_data)
{
  FILE* file = reinterpret_cast<FILE*>(user_data);
  fflush(file);
}

inline const char* filename(const char* path)
{
  for (auto ptr = path; *ptr; ++ptr) {
      if (*ptr == '/' || *ptr == '\\') {
          path = ptr + 1;
      }
  }
  return path;
}

inline void log(VerBoseType verbose, const char* file, unsigned line, const char* format, ...)
{
  va_list vlist;
  va_start(vlist, format);
  auto buff = vtextprintf(format, vlist);
  log_to_all(2, verbose, file, line, "", buff.c_str());
  va_end(vlist);
}

inline void log_and_abort(int stack_trace_skip, const char* expr, const char* file, unsigned line, const char* format, ...)
{
  va_list vlist;
  va_start(vlist, format);
  auto buff = vtextprintf(format, vlist);
  log_to_all(stack_trace_skip + 1, Verbose_FATAL, file, line, expr, buff.c_str()); // will invoke the abort().
  va_end(vlist);
}

inline void log_to_all(int  stack_trace_skip,
                       VerBoseType    verbose,
                       const char*    file,
                       unsigned       line,
                       const char*    prefix,
                       const char*    buff)
{
  char preamble_buff[128];
  print_preamble(preamble_buff, sizeof(preamble_buff), verbose, file, line);
  auto message = Message{verbose, file, line, preamble_buff, prefix, buff};
  log_message(stack_trace_skip + 1, message, true);
}

inline text get_error_context()
{
  return get_error_context_for(LoggerConfig::pthreadErrCtPtr);
}
inline text get_error_context_for(const ErrContext* head)
{
  std::vector<const ErrContext*> stack;
  while (head) {
      stack.push_back(head);
      head = head->previous_;
  }
  std::reverse(stack.begin(), stack.end());

  std::string result;
  if (!stack.empty()) {
      result += "------------------------------------------------\n";
      for (auto entry : stack) {
          const auto description = std::string(entry->descr_) + ":";
          auto prefix = textprintf("[ErrorContext] %*s:%-5u %-20s ",
                                   LoggerConfig::filenNameWidth, filename(entry->file_),
                                   entry->line_, description.c_str());
          result += prefix.c_str();
          //entry->print_value(result); /// ??? meaning??
          result += "\n";
      }
      result += "------------------------------------------------";
  }
  return text(strdup(result.c_str()));
}

#if STACKTRACES
inline void do_replacements(const PairList& replacements, std::string& str)
{
  for(auto&& pair:replacements)
  {
      if (pair.first.size() <= pair.second.size())
      {
        // On gcc, "type_name<std::string>()" is "std::string"
        continue;
      }
      size_t it;
      while ((it=str.find(pair.first)) != std::string::npos)  // find first
      {
        str.replace(it, pair.first.size(), pair.second);
      }
  }
}
inline std::string set_friendly_stacktrace(const std::string& input)
{
  std::string output = input;

  //do_replacements(s_user_stack_cleanups, output); // ????? s_user_stack_cleanups
  do_replacements(replaceList, output);

  try {
    std::regex std_allocator_re(R"(,\s*std::allocator<[^<>]+>)");
    output = std::regex_replace(output, std_allocator_re, std::string(""));
    std::regex template_spaces_re(R"(<\s*([^<> ]+)\s*>)");
    output = std::regex_replace(output, template_spaces_re, std::string("<$1>"));
  } catch (std::regex_error&) {/*may throw exception*/}

  return output;
}

// we use libstdc++'s abi::__cxa_demangle() to set friendly demangled stack trace.
// Example:
//
// | Mangled Name  | abi::__cxa_demangle()
// |---------------|-----------------------
// | _Z1fv         | f()
// | _Z1fi         | f(int)
// | _Z3foo3bar    | foo(bar)
// | _Z1fIiEvi     | void f<int>(int)
// | _ZN1N1fE      | N::f
// | _ZN3Foo3BarEv | Foo::Bar()
// | _Zrm1XS_"     | operator%(X, X)
// | _ZN3FooC1Ev   | Foo::Foo()
// | _Z1fSs        | f(std::basic_string<char,
// |               |   std::char_traits<char>,
// |               |   std::allocator<char> >)
inline std::string stacktrace_as_stdstring(int skip)
{
  void* callstack[256];
  const auto max_frames = sizeof(callstack) / sizeof(callstack[0]);
  int num_frames = backtrace(callstack, max_frames);
  char** symbols = backtrace_symbols(callstack, num_frames);

  std::string result;
  // reverse the stack trace result	
  for (int i = num_frames - 1; i >= skip; --i) {
  	char buf[1024];

   	// @brief typedef struct {
   	//          const char *dli_fname;  /* Pathname of shared object that contains address */
   	//          void       *dli_fbase;  /* Address at which shared object is loaded */
   	//          const char *dli_sname;  /* Name of nearest symbol with address lower than addr */
   	//          void       *dli_saddr;  /* Exact address of symbol named in dli_sname */
   	//         } Dl_info;
   	// If no symbol matching addr could be found, then dli_sname and dli_saddr are set to NULL.
   	// The function dladdr() takes a function pointer and tries to resolve name and file where it is located
   	Dl_info info;
   	if (dladdr(callstack[i], &info) && info.dli_sname) {
      	    char* demangled = NULL;
      	    int status = -1;
      	    if (info.dli_sname[0] == '_') {
        	    demangled = abi::__cxa_demangle(info.dli_sname, 0, 0, &status);
      	    }
      	    snprintf(buf, sizeof(buf), " %-3d %*p %s + %zd\n",
             	i - skip, int(2 + sizeof(void*) * 2), callstack[i],
             	status == 0 ? demangled :
             	info.dli_sname == 0 ? symbols[i] : info.dli_sname,
             	static_cast<char*>(callstack[i]) - static_cast<char*>(info.dli_saddr));
      	    free(demangled);
  	} else {
      	    snprintf(buf, sizeof(buf), "%-3d %*p %s\n",i - skip, int(2 + sizeof(void*) * 2), callstack[i], symbols[i]);
   	}
   	result += buf;
  }//for
  free(symbols);

  if (num_frames == max_frames) {
  	result = "[logger truncated]\n" + result;
  }

  if (!result.empty() && result[result.size() - 1] == '\n') {
  	result.resize(result.size() - 1);
  }

  return set_friendly_stacktrace(result);
}
#else //STACKTRACES
inline void do_replacements(const PairList& replacements, std::string& str){}
inline std::string set_friendly_stacktrace(const std::string& input){return "";}
inline std::string stacktrace_as_stdstring(int)
{
    #warning "Logger warning: No stacktraces available on this platform [ DISABLED ]"
    return "";
}
#endif //STACKTRACES

inline text stacktrace(int skip)
{
  auto str = stacktrace_as_stdstring(skip + 1);
  return text(strdup(str.c_str()));
}

inline void set_thread_name(const char *name)
{
  #ifdef LOGGER_TLS_NAMES
  LoggerConfig::pthreadKeyName = strdup(name);  // name must call free(LoggerConfig::pthreadKeyName) to release the mem in the end
  #else
    #ifdef __APPLE__
       pthread_setname_np(name);
    #else
       pthread_setname_np(pthread_self(), name);
    #endif
  #endif
}

inline void get_thread_name(char* buffer, unsigned long long length, bool right_align_hext_id)
{
  if(length == 0u){
	IN_LOG(ERROR,"get_thread_name get 0 length buffer. ");
  }
  if(buffer == nullptr){
	IN_LOG(ERROR,"get_thread_name get nullptr buffer. ");
  }
  #if SUPPORT_PTHREADS
    auto thread = pthread_self();
    #if LOGGER_TLS_NAMES
      if (const char* name = LoggerConfig::pthreadKeyName) {
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
        snprintf(buffer, length, "%*X", static_cast<int>(length - 1), static_cast<unsigned>(thread_id));
      } else {
        snprintf(buffer, length, "%X", static_cast<unsigned>(thread_id));
      }
    }
  #else // SUPPORT_PTHREADS
    buffer[0] = 0;
  #endif // SUPPORT_PTHREADS
}

inline text vtextprintf(const char* format, va_list vlist)
{
  char* buff = nullptr;
  int result = vasprintf(&buff, format, vlist);
  if(result == -1){
	IN_LOG(ERROR,"Bad string format: '%s'\n", format);
    //fprintf(stderr,"Bad string format: '%s'\n", format);
    //fflush(stderr);
  }
  //CHECK_F(result ==-1, "Bad string format: '%s'", format);
  return text(buff);
}

inline text textprintf(const char* format, ...)
{
  va_list vlist;
  va_start(vlist, format);
  auto result = vtextprintf(format, vlist);
  va_end(vlist);
  return result;
}

inline void print_preamble(char* out_buff,
                    size_t out_buff_size,
                    VerBoseType    verbose,
                    const char* file,
                    unsigned line)
{
  long long ms_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  time_t sec_since_epoch = time_t(ms_since_epoch / 1000);
  tm time_info;
  localtime_r(&sec_since_epoch, &time_info);

  auto uptime_ms = duration_cast<milliseconds>(steady_clock::now() - startTime).count();
  auto uptime_sec = uptime_ms / 1000.0;

  char thread_name[LoggerConfig::threadNameWidth + 1] = {0};
  get_thread_name(thread_name, LoggerConfig::threadNameWidth + 1, true);

  if (LoggerConfig::splitFileName) {
    file = filename(file);
  }

  char level_buff[6];
  if (verbose <= Verbose_FATAL) {
    snprintf(level_buff, sizeof(level_buff) - 1, "FTL");
  } else if (verbose == Verbose_ERROR) {
    snprintf(level_buff, sizeof(level_buff) - 1, "ERR");
  } else if (verbose == Verbose_WARNING) {
    snprintf(level_buff, sizeof(level_buff) - 1, "WAN");
  } else {
    snprintf(level_buff, sizeof(level_buff) - 1, "%3d", verbose);
  }

  snprintf(out_buff, out_buff_size, "%4s| %02d:%02d:%02d.%05lld| %.3fs| %s| %s:%u] ",
		   level_buff,
           time_info.tm_hour, time_info.tm_min, time_info.tm_sec, ms_since_epoch % 1000,
           uptime_sec,
           thread_name,
           file, 
           line);

  /*snprintf(out_buff, out_buff_size, "%04d-%02d-%02d %02d:%02d:%02d.%05lld (%8.3fs) [%-*s]%*s:%-5u %4s| ",
           1900 + time_info.tm_year, 1 + time_info.tm_mon, time_info.tm_mday,
           time_info.tm_hour, time_info.tm_min, time_info.tm_sec, ms_since_epoch % 1000,
           uptime_sec,
           LoggerConfig::threadNameWidth, thread_name,
           LoggerConfig::filenNameWidth,file, 
	   line, level_buff);*/
}

inline void log_message(int stack_trace_skip, Message& message, bool abort_if_fatal)
{
  const auto verbosity = message.verbose;
  std::lock_guard<std::recursive_mutex> lock(LoggerConfig::rsMutex);

  if (verbosity <= LoggerConfig::currentVerbos) {
    if (LoggerConfig::colorstderr && LoggerConfig::terminalSupportColor) {
      if (verbosity > Verbose_WARNING) {
        fprintf(stderr, "%s%s%s%s%s%s%s%s%s\n",
                reset(),
                dim(),
                message.preamble,
                reset(),
				bold(),
                verbosity == Verbose_INFO ? bold() : light_gray(),
                message.prefix,
                message.message,
                reset());
      } else {
        fprintf(stderr, "%s%s%s%s%s%s%s\n",
        reset(),
        bold(),
		verbosity == Verbose_WARNING ? yellow() : (verbosity == Verbose_FATAL ? b_red() : red()),
		message.preamble,
        message.prefix,
        message.message,
        reset());
      }
    } else {
      fprintf(stderr, "%s%s%s\n",
      message.preamble, message.prefix, message.message);
    }

    if (LoggerConfig::flushBufferInMs == 0) {
      fflush(stderr);
    } else {
      LoggerConfig::needFlush = true;
    }
  } // if verbosity <= LoggerConfig::currentVerbos

  if (verbosity == Verbose_FATAL) {
    auto st = stacktrace(stack_trace_skip + 2); // friendly message of stack trace
    if (!st.empty()) {
      	if(LoggerConfig::colorstderr && LoggerConfig::terminalSupportColor) {
      		fprintf(stderr," %s%s%s*** %s fatal error: stack trace: ***\n %s \n %s", 
      					reset(),
      					bold(),
      					red(),
						message.prefix,
      					st.c_str(),
      					reset());
		} else {
			fprintf(stderr," *** %s fatal error stack trace: ***:\n %s \n",message.prefix,st.c_str());
		}
		fflush(stderr);
    }

    auto ec = get_error_context(); // new start 2016/12/4
    if (!ec.empty()) {
        //fprintf(stderr,"error %s\n", ec.c_str());
        //fflush(stderr);
        IN_LOG(ERROR, "%s", ec.c_str());
    }
  }



  for (auto& p : LoggerConfig::callbackVecs) {
    if (verbosity <= p.verbose) {
      p.callback(p.user_data, message); // log to file
      if (LoggerConfig::flushBufferInMs == 0) {
        // fflush(file)
        if (p.flush) { p.flush(p.user_data); }
      } else {
        LoggerConfig::needFlush = true;
      }
    }
  }

  if (LoggerConfig::flushBufferInMs > 0 && !LoggerConfig::flushThread) {
    // create the guard thread preiodic flushing
     LoggerConfig::flushThread = new std::thread([](){
      for (;;) {
          if (LoggerConfig::needFlush) {
            // if flushBufferInMs >0 flush everything!
            flush_callback();
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(LoggerConfig::flushBufferInMs));
        }
    });
  }

  if (verbosity == Verbose_FATAL) {
    flush_callback();

    /*if (s_fatal_handler) {
        s_fatal_handler(message);
        flush_callback();
    }*/

    if (abort_if_fatal) {
  #if LOGGER_CATCH_SIGABRT && !defined(_WIN32)
      // Make sure we don't catch our own abort:
      signal(SIGABRT, SIG_DFL);
  #endif
      abort();
    }
  }
}

inline void write_to_stderr(const char* data, size_t size) {
  auto result = write(STDERR_FILENO, data, size);
  (void)result; // Ignore errors.
}

inline void write_to_stderr(const char* data) {
  write_to_stderr(data, strlen(data));
}

inline void call_default_signal_handler(int signal_number) {
  struct sigaction sig_action;
  memset(&sig_action, 0, sizeof(sig_action));
  sigemptyset(&sig_action.sa_mask);
  sig_action.sa_handler = SIG_DFL; // set sig_del for default signal handle
  sigaction(signal_number, &sig_action, NULL);
  // send signal to a process . not kill the pthread
  kill(getpid(), signal_number);
}

inline void logger_signal_handler(int signal_number, siginfo_t*, void*){
  const char* signal_name = "UNKNOWN SIGNAL";

  for (const auto& s : LOCAL_SIGS) {
       if (s.sigNum == signal_number) {
          signal_name = s.sigName;
          break;
       }
  }
  // thread safety
  if (LoggerConfig::terminalSupportColor && LoggerConfig::colorstderr) {
      write_to_stderr(reset());
      write_to_stderr(bold());
      write_to_stderr(blink());
      write_to_stderr(light_red());
  }
  write_to_stderr("\n");
  write_to_stderr("logger caught a signal: ");
  write_to_stderr(signal_name);
  write_to_stderr("\n");
  if (LoggerConfig::terminalSupportColor && LoggerConfig::colorstderr) {
    write_to_stderr(reset());
  }
  // unsafe things
  flush_callback();
  char preamble_buff[128];
  print_preamble(preamble_buff, sizeof(preamble_buff), Verbose_FATAL, "", 0);
  auto message = Message{Verbose_FATAL, "", 0, preamble_buff, "Signal caught: ", signal_name};
  try {
    log_message(2, message, false); // may throw some runtime exception to disable signal handler, set false to use our defined signal func.
  } catch (...){write_to_stderr("Exception caught and ignored by logger signal handler.\n");}

  call_default_signal_handler(signal_number);
}

inline void install_logger_signal_handlers() {
  struct sigaction sig_action;
  memset(&sig_action, 0, sizeof(sig_action));
  sigemptyset(&sig_action.sa_mask);
  sig_action.sa_flags |= SA_SIGINFO;
  sig_action.sa_sigaction = &logger_signal_handler;
  for (const auto& s : LOCAL_SIGS) {
    if(sigaction(s.sigNum, &sig_action, NULL) == -1){
        fprintf(stderr,"Failed to install handler for %s\n", s.sigName);
    }
  }
}
/**
 *  \brief logger init func
 *
 */
inline void initial(const char* argv0){
  LoggerConfig::init();
  // get host and user info.
  get_username();
  get_program_name(argv0);
  get_hostname();
#if LOGGER_TLS_NAMES
  set_thread_name("main_thread"); // set main thread name = "main_thread" 
#endif
  if(!LoggerConfig::logtostderr) {
  	add_sys_log_file("./log");
  }
  auto PREAMBLE_EXPLAIN = textprintf("  v  |     time     |  uptime  | %s | %s:line ] ", " thread name/id", "file");
  if (LoggerConfig::currentVerbos >= Verbose_INFO) {
    if (LoggerConfig::colorstderr && LoggerConfig::terminalSupportColor) {
      fprintf(stderr, "%s%s%s\n", reset(), dim(), PREAMBLE_EXPLAIN.c_str());
    } else {
      fprintf(stderr, "%s", PREAMBLE_EXPLAIN.c_str());
    }
    fflush(stderr);
  }
  IN_LOG(INFO,"Current Verbose Level: %d", LoggerConfig::currentVerbos);
  IN_LOG(INFO,"-----------start logging------------"); 
  //fprintf(stderr,"stderr verbosity: %d\n", LoggerConfig::currentVerbos);
  //fprintf(stderr,"-----------------------------------\n");
  //fflush(stderr);
  install_logger_signal_handlers();
  flush_callback();

}


} // namespace funcRegister

inline loggerMsg::~loggerMsg()
{
  auto message = ss_.str();
  if(isAbort_){
    funcRegister::log_and_abort(1, expression_, file_, line_, "%s", message.c_str());
  }else{
    funcRegister::log(verbose_, file_, line_, "%s", message.c_str());
  }
}

} // namespace core

} // namespace logger


#endif // LOGGER_CORE_H
