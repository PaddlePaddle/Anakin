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
#ifndef ANAKIN_SABER_CORE_IMPL_AMD_UTILS_AMDERROR_H
#define ANAKIN_SABER_CORE_IMPL_AMD_UTILS_AMDERROR_H

#include <exception>
#include <iostream>
#include <string>
#include <tuple>

namespace anakin {
namespace saber {

struct Exception : std::exception {
    std::string message;
    int status;
    Exception(const std::string& msg = "") : message(msg), status(-1) {}

    Exception(int s, const std::string& msg = "") : message(msg), status(s) {}

    Exception SetMessage(const std::string& file, int line) {
        message = file + ":" + std::to_string(line) + ": " + message;
        return *this;
    }

    const char* what() const noexcept override {
        return message.c_str();
    }
};

#define AMD_THROW(...) throw anakin::saber::Exception(__VA_ARGS__).SetMessage(__FILE__, __LINE__)

} // namespace saber
} // namespace anakin

#endif
