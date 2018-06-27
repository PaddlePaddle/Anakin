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
#ifndef ANAKIN_LOGGER_CORE_H
#define ANAKIN_LOGGER_CORE_H

#include "logger_utils.h"

namespace logger {

namespace core {

/**
 *  \brief logger init func
 *
 */
inline void initial(const char* argv0){ 
    utils::sys::init(); 
    utils::sys::set_max_logger_verbose_level(Verbose_Max);
    // get host and user info.  
    utils::sys::get_username(); 
    utils::sys::get_program_name(argv0); 
    utils::sys::get_hostname(); 
#if LOGGER_TLS_NAMES 
    utils::sys::set_thread_name("main_thread"); // set main thread name = "main_thread" 
#endif 
    // log to file
    utils::sys::add_sys_log_file("./log"); 
    utils::sys::set_file_barrier_size_in_GB(10);

    
    utils::sys::install_logger_signal_handlers();
    utils::sys::file_flush_all();
}

/**
 * \brief dispatch message class
 */
template<VerBoseType Verbose>
class LoggerDispatchMsg {
public:
    void operator()(const char* expression, const char* file, unsigned line, const char* msg) {
        log_msg_dispatch(expression, file, line, msg);
    }

private:
    /// compose the target log msg with non-Fatal log
    /// note: 
    ///    this api can accept c-format(printf) input msg
    inline void log_msg_dispatch(const char* expr, 
								 const char* file, 
								 unsigned line, 
								 const char* msg) {
		char header[128];
        utils::sys::add_log_header<Verbose>(header, sizeof(header), file, line);
		send_msg(false, expr, header, msg, DevType<__ERR>());
		send_msg(false, expr, header, msg, DevType<__FILE>());
    }

	inline void send_msg(bool exception, const char* expr, const char* header, const char* msg, DevType<__ERR>) {
  		std::lock_guard<std::recursive_mutex> lock(utils::SYSResource::Global().mut);
  		if (Verbose <= utils::sys::get_max_logger_verbose_level()) {
  		    if (utils::sys::colorful_shell_access()) {
  		        if (Verbose > Verbose_WARNING) {
  		            fprintf(stderr, "%s%s%s%s%s%s%s%s%s\n",
  		                    Color<RESET>::str,
  		                    Color<DIM>::str,
  		                    header,
  		                    Color<RESET>::str,
  		                    Color<BOLD>::str,
  		                    Verbose == Verbose_INFO ? Color<BOLD>::str : Color<LIGHT_GRAY>::str,
  		                    expr,
  		                    msg,
  		                    Color<RESET>::str);
  		        } else {
  		          fprintf(stderr, "%s%s%s%s%s%s%s\n",
  		                  Color<RESET>::str,
  		                  Color<BOLD>::str,
  		                  Verbose == Verbose_WARNING ? Color<YELLOW>::str : \
                          (Verbose == Verbose_FATAL ? Color<BOLD_RED>::str : Color<RED>::str),
  		                  header,
  		                  expr,
  		                  msg,
  		                  Color<RESET>::str);
  		        }
  		    } else {
  		        fprintf(stderr, "%s%s%s\n", header, expr, msg);
  		    }

            // here use file flush to flush everything
            //fflush(stderr);
            utils::sys::file_flush_all();
  		} // if Verbose <= LoggerConfig::currentVerbos	

		if (exception) {
    		_stack_trace_inf = utils::sys::stack_trace(5); 
    		if (!_stack_trace_inf.empty()) {
    	    	if (utils::sys::colorful_shell_access()) {
    	        	fprintf(stderr," %s%s%s >>>>> Fatal error: stack trace: <<<<<< \n %s \n %s\n",
    	                    Color<RESET>::str,
    	                    Color<BOLD>::str,
    	                    Color<GREEN>::str,
    	                    _stack_trace_inf.c_str(),
    	                    Color<RESET>::str);
    	    	} else {
    	        	fprintf(stderr,"  >>>>> Fatal error: stack trace: <<<<<< \n %s \n", _stack_trace_inf.c_str());
    	    	}
    	    	fflush(stderr);
    		}
		}
	}

	inline void send_msg(bool exception, const char* expr, const char* header, const char* msg, DevType<__FILE>) {
        utils::sys::write_file_log(Verbose, expr, header, msg);

  		if (exception) {
            utils::sys::write_file_log(Verbose, " >>>>> Fatal error: stack trace: <<<<< \n ", "", _stack_trace_inf.c_str());
            utils::sys::file_flush_target(Verbose);

    		if (exception) {
 #if LOGGER_CATCH_SIGABRT && !defined(_WIN32) 
                // Make sure we don't catch our own abort: 
                signal(SIGABRT, SIG_DFL);
 #endif
      			abort();
    		}
  		}
	}

private:
    std::string _stack_trace_inf{""};
};

template<>
inline void LoggerDispatchMsg<Verbose_FATAL>::log_msg_dispatch(const char* expr, 
															   const char* file, 
															   unsigned line, 
															   const char* msg) {
	char header[128]; 
    utils::sys::add_log_header<Verbose_FATAL>(header, sizeof(header), file, line);
	send_msg(true, expr, header, msg, DevType<__ERR>());
	send_msg(true, expr, header, msg, DevType<__FILE>());
}


/**
 * \brief logger central class
 */
template<VerBoseType Verbose>
class LoggerMsg {
public:
    LoggerMsg(const char* file, unsigned line)
        :_file(file), _line(line) {}
    LoggerMsg(const char* expression, const char* file, unsigned line)
        :_expression(expression), _file(file), _line(line) {}

    ~LoggerMsg() {
        const char* msg = std::string(_msg.str()).c_str();
        this->_dispatch(_expression.c_str(), _file.c_str(), _line, msg);
    }

    template<typename T> 
    LoggerMsg& operator<<(const T& var) {
        _msg << var;
        return *this;
    }

    /// access for std::endl and other std io
    LoggerMsg& operator<<(std::ostream&(*func)(std::ostream&)) {
        func(_msg);
        return *this;
    }

private:
    std::string _expression;
    std::string _file;
    unsigned _line;
    std::ostringstream _msg;
    LoggerDispatchMsg<Verbose> _dispatch;
};

/// turn class LoggerMsg instance into void function, it's a little trick.
class voidify { 
public: 
    voidify(){} 
    template<VerBoseType Verbose>
    void operator&(const LoggerMsg<Verbose>&){} 
};

} /* namespace core */

} /* namespace logger */

#endif
