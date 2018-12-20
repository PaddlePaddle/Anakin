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
#ifndef ANAKIN_SABER_FUNCS_IMPL_UTILS_AMDFILEUTILS_H
#define ANAKIN_SABER_FUNCS_IMPL_UTILS_AMDFILEUTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <openssl/md5.h>
#include <miopen/db.hpp>

namespace anakin {
namespace saber {

std::string temp_directory_path();
bool is_directory(const std::string& path);
std::string filename(const std::string& filename);
std::string remove_filename(std::string path);
int permissions(std::string& p, mode_t mode);
std::string unique_path();
bool exists(std::string path);
std::string parent_path(const std::string& path);
bool create_directories(std::string path);

std::string
GetCacheFile(const std::string& device, const std::string& name, const std::string& args);

std::string GetCachePath();

void SaveBinary(
    const std::string& binary_path,
    const std::string& device,
    const std::string& name,
    const std::string& args);

std::string
LoadBinaryPath(const std::string& device, const std::string& name, const std::string& args);

std::string LoadFile(const std::string& s);
std::string md5(std::string s);
miopen::Db GetDb(std::string device_name, int max_CU);
std::string genTempFilePath(std::string name);
void writeFile(const std::string& content, const std::string& name);

} // namespace saber
} // namespace anakin
#endif
