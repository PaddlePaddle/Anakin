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

#ifndef ANAKIN_BASE_H
#define ANAKIN_BASE_H 

#include <string>
#include <utility>
#include <type_traits>
#include "framework/core/types.h"

namespace anakin {

/// This class is used to return the status of functions.
class Status {
public:
    Status():_is_suc(RetType::SUC), _error_msg("") {}
    Status(RetType ret):_is_suc(ret), _error_msg("") {}
    Status(RetType ret, const char* err_msg ="Not known"):_is_suc(ret), _error_msg(err_msg) {}

    static Status OK(const char* msg = "") { return Status{RetType::SUC, msg}; }
    static Status ANAKINFAIL(const char* msg = "Not known") { return Status{RetType::ERR, msg}; }
    static Status EXIT(const char* msg = "succeessfully exit") { return Status{RetType::IMME_EXIT, msg}; }

    operator bool() const { return (_is_suc == RetType::SUC) || (_is_suc == RetType::IMME_EXIT); }

    const char* info() const { return _error_msg.c_str(); }

    bool operator==(const Status& status);
    bool operator!=(const Status& status);

    /// copy and move
    Status(const Status& status);
    Status(const Status&& status);
    Status& operator=(const Status& status);
    Status& operator=(const Status&& status);

private:
    std::string _error_msg; 
    RetType _is_suc{RetType::SUC};
};

inline bool Status::operator==(const Status& status) {
    return (this->_is_suc == status._is_suc);
}

inline bool Status::operator!=(const Status& status) {
    return (this->_is_suc != status._is_suc);
}

inline Status::Status(const Status& status) {
    this->_error_msg = status._error_msg;
    this->_is_suc = status._is_suc;
}

inline Status::Status(const Status&& status) {
    this->_error_msg = std::move(status._error_msg);
    this->_is_suc = status._is_suc;
}

inline Status& Status::operator=(const Status& status) {
    this->_error_msg = status._error_msg;
    this->_is_suc = status._is_suc;
    return *(this);
}

inline Status& Status::operator=(const Status&& status) {
    this->_error_msg = std::move(status._error_msg);
    this->_is_suc = status._is_suc;
    return *(this);
}

} /* namespace anakin */

#endif
