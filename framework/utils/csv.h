#ifndef ANAKIN_FRAMEWORK_UTILS_CSV_H
#define ANAKIN_FRAMEWORK_UTILS_CSV_H

#include <iostream>
#include <fstream>

#ifdef ENABLE_OP_TIMER

namespace anakin {

class Csvfile;

inline static Csvfile& endrow(Csvfile& file);
inline static Csvfile& flush(Csvfile& file);

class Csvfile {

public:
    Csvfile(std::string const& file, bool app_mode = false, \
    std::string const& sep = ",")
        : _fs()
        , _is_first(true)
        , _sep(sep)
        , _esc("\"")
        , _special_chars("\"") {
        _fs.exceptions(std::ios::failbit | std::ios::badbit);
        if (app_mode) {
            _fs.open(file, std::ofstream::app);
        } else {
            _fs.open(file);
        }
    }

    ~Csvfile() {
        flush();
        _fs.close();
    }

    void flush() {
        _fs.flush();
    }

    void endrow() {
        _fs << std::endl;
        _is_first = true;
    }

    Csvfile& operator << (Csvfile& (*func)(Csvfile&)) {
        return func(*this);
    }

    template<typename T>
    Csvfile& operator << (const T& val) {
        return write(val);
    }

    Csvfile& operator << (const char* val) {
        return write(escape(val));
    }

    Csvfile& operator << (const std::string& val) {
        return write(escape(val));
    }

private:
    std::ofstream _fs;
    bool _is_first;
    const std::string _sep;
    const std::string _esc;
    const std::string _special_chars;

    template<typename T>
    Csvfile& write(const T& val) {
        if (!_is_first) {
            _fs << _sep;
        } else {
            _is_first = false;
        }
        _fs << val;
        return *this;
    }

    std::string escape(const std::string & val) {
        std::ostringstream result;
        result << '"';
        std::string::size_type to, from = 0u, len = val.length();
        while (from < len && \
            std::string::npos != (to = val.find_first_of(_special_chars, from))) {
            result << val.substr(from, to - from) << _esc << val[to];
            from = to + 1;
        }
        result << val.substr(from) << '"';
        return result.str();
    }
};

inline static Csvfile& endrow(Csvfile& file) {
    file.endrow();
    return file;
}

inline static Csvfile& flush(Csvfile& file) {
    file.flush();
    return file;
}

}

#endif /* ENABLE_OP_TIMER */

#endif /* ANAKIN_FRAMEWORK_UTILS_CSV_H */
