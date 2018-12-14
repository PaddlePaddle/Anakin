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
#include "amd_file_utils.h"
#include <miopen/db_path.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "amd_logger.h"
#include "amd_error.h"
#include <vector>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace anakin {
namespace saber {

#define SABER_CACHE_DIR "~/.cache/amd_saber/"
#define separator "/"

std::string md5(std::string s) {
    std::array<unsigned char, MD5_DIGEST_LENGTH> result {};
    MD5(reinterpret_cast<const unsigned char*>(s.data()), s.length(), result.data());

    std::ostringstream sout;
    sout << std::hex << std::setfill('0');

    for (auto c : result)
        sout << std::setw(2) << int {c};

    return sout.str();
}

inline std::string
ReplaceString(std::string subject, const std::string& search, const std::string& replace) {
    size_t pos = 0;

    while ((pos = subject.find(search, pos)) != std::string::npos) {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    return subject;
}

bool is_directory_separator(char c) {
    return c == '/';
}

bool is_root_separator(const char* str, size_t pos) {
    // pos is position of the separator
    if (str != nullptr && !is_directory_separator(str[pos])) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "precondition violation";
    }

    // subsequent logic expects pos to be for leftmost slash of a set
    while (pos > 0 && is_directory_separator(str[pos - 1])) {
        --pos;
    }

    //  "/" [...]
    if (pos == 0) {
        return true;
    }

    //  "//" name "/"
    if (pos < 3 || !is_directory_separator(str[0]) || !is_directory_separator(str[1])) {
        return false;
    }

    std::string tmp(str);
    return tmp.find_first_of("/", 2) == pos;
}

size_t root_directory_start(const char* path, size_t size) {
    // return npos if no root_directory found
    // case "//"
    if (size == 2 && is_directory_separator(path[0]) && is_directory_separator(path[1])) {
        return std::string::npos;
    }

    // case "//net {/}"
    if (size > 3 && is_directory_separator(path[0]) && is_directory_separator(path[1])
            && !is_directory_separator(path[2])) {
        std::string str(path);
        size_t pos = str.find_first_of("/", 2);
        return pos < size ? pos : std::string::npos;
    }

    // case "/"
    if (size > 0 && is_directory_separator(path[0])) {
        return 0;
    }

    return std::string::npos;
}

size_t filename_pos(const char* str, size_t end_pos) {
    // end_pos is past-the-end position
    // return 0 if str itself is filename (or empty)

    // case: "//"
    if (end_pos == 2 && is_directory_separator(str[0]) && is_directory_separator(str[1])) {
        return 0;
    }

    // case: ends in "/"
    if (end_pos && is_directory_separator(str[end_pos - 1])) {
        return end_pos - 1;
    }

    // set pos to start of last element
    std::string filename(str);
    size_t pos = (filename.find_last_of('/', end_pos - 1));

    return (pos == std::string::npos // path itself must be a filename (or empty)
            || (pos == 1 && is_directory_separator(str[0]))) // or net
           ? 0                                       // so filename is entire string
           : pos + 1;                                // or starts after delimiter
}

size_t parent_path_end(const std::string& path) {
    size_t end_pos = filename_pos(path.c_str(), path.length());

    bool filename_was_separator = path.length() && is_directory_separator(path.c_str()[end_pos]);

    // skip separators unless root directory
    size_t root_dir_pos = root_directory_start(path.c_str(), end_pos);

    for (; end_pos > 0 && (end_pos - 1) != root_dir_pos
            && is_directory_separator(path.c_str()[end_pos - 1]);
            --end_pos) {
    }

    return (end_pos == 1 && root_dir_pos == 0 && filename_was_separator) ? std::string::npos
           : end_pos;
}

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }

    if (pos1 != s.length()) {
        v.push_back(s.substr(pos1));
    }
}

std::string temp_directory_path() {
    const char* val = 0;
    (val = std::getenv("TMPDIR")) || (val = std::getenv("TMP")) || (val = std::getenv("TEMP"))
    || (val = std::getenv("TEMPDIR"));

    const char* default_tmp = "/tmp";
    std::string temp_directory((val != 0) ? val : default_tmp);
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "temp directory path :" << temp_directory;
    return temp_directory;
}

bool is_directory(const std::string& path) {
    struct stat info;

    if (stat(path.c_str(), &info) == 0) {
        if (info.st_mode & S_IFDIR) {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "it's a directory";
            return true;
        } else if (info.st_mode & S_IFREG) {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "it's a file";
            return false;
        } else {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "not file or directory";
        }
    } else {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "path not exist";
    }

    return false;
}

std::string filename(const std::string& filename) {
    size_t pos = filename_pos(filename.c_str(), filename.length());
    return (filename.length() && pos && is_directory_separator(filename.c_str()[pos])
            && !is_root_separator(filename.c_str(), pos))
           ? "."
           : filename.substr(pos, filename.length());
}

std::string remove_filename(std::string path) {
    return path.erase(parent_path_end(path), path.length());
}

int permissions(std::string& p, mode_t mode) {
    int result = chmod(p.c_str(), mode);

    if (result != 0) {
        result = errno;
    }

    return (result);
}

std::string unique_path() {
    int unique_path_len = 19;
    std::string str     = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::string newstr;
    int pos, i = 0;
    std::string str1 = "-";

    while (newstr.size() != unique_path_len) {
        pos = ((rand() % (str.size() - 1)));
        newstr += str.substr(pos, 1);
        i++;

        if (i == 4 && newstr.size() != unique_path_len) {
            newstr += "-";
            i = 0;
        }
    }

    return newstr;
}

bool exists(std::string path) {
    int state = access(path.c_str(), R_OK | W_OK);

    if (state == 0) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "file path exist";
        return true;
    } else {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "file path not exist, path :" << path;
        return false;
    }
}

std::string parent_path(const std::string& path) {
    size_t end_pos = parent_path_end(path);
    return end_pos == std::string::npos ? path : path.substr(0, end_pos);
}

int file_status(const std::string& path) {
    struct stat info;

    if (stat(path.c_str(), &info) == 0) {
        if (info.st_mode & S_IFDIR) {
            return 1; // it's a directory
        } else if (info.st_mode & S_IFREG) {
            return 2; // it's a file
        } else {
            return 3; // not direcort or file
        }
    } else {
        return 3;
    }
}

bool filename_is_dot(std::string p) {
    std::string name = filename(p);
    return name.length() == 1 && *name.c_str() == '.';
}

bool filename_is_dot_dot(std::string p) {
    return p.length() >= 2 && p.c_str()[p.length() - 1] == '.' && p.c_str()[p.length() - 2] == '.';
}

bool create_directory(const std::string& p) {
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "path :" << p;
    int flag = mkdir(p.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

    if (flag == 0) {
        return true;
    }

    // attempt to create directory failed
    if (is_directory(p)) {
        return false;
    }

    //  attempt to create directory failed && it doesn't already exist
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) <<
                                         "attempt to create directory failed && it doesn't already exist";
    return false;
}

bool create_directories(const std::string p) {
    if (p.length() == 0) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "invalid argument for path";
        return false;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "path :" << p;

    if (filename_is_dot(p) || filename_is_dot_dot(p)) {
        return create_directories(parent_path(p));
    }

    if (is_directory(p)) {
        return false;
    }

    std::string parent = parent_path(p);

    if (parent == p) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "internal error: path == parent path";
    }

    if (parent.length() > 0) {
        // if the parent does not exist, create the parent
        if (!is_directory(p)) {
            create_directories(parent);
        }
    }

    // create the directory
    return create_directory(p);
}

// original functions
std::string ComputeCachePath() {
    std::string cache_dir = SABER_CACHE_DIR;

    auto p = ReplaceString(cache_dir, "~", getenv("HOME"));

    if (!exists(p)) {
        create_directories(p);
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "Get program path :" << p;

    return p;
}

std::string GetCachePath() {
    static const std::string path = ComputeCachePath();
    return path;
}

std::string
GetCacheFile(const std::string& device, const std::string& name, const std::string& args) {
    std::string filename = name + ".o";
    return GetCachePath() + md5(device + ":" + args) + "/" + filename;
}

std::string
LoadBinaryPath(const std::string& device, const std::string& name, const std::string& args) {
    auto f = GetCacheFile(device, name, args);

    if (exists(f)) {
        return f;
    } else {
        return {};
    }
}

void SaveBinary(
    const std::string& binary_path,
    const std::string& device,
    const std::string& name,
    const std::string& args) {
    auto p = GetCacheFile(device, name, args);
    create_directories(parent_path(p));
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "target path :" << p;
    rename(binary_path.c_str(), p.c_str());
}

std::string LoadFile(const std::string& s) {
    std::ifstream t(s);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

miopen::Db GetDb(std::string device_name, int max_CU) {
    auto p = ReplaceString(miopen::GetDbPath(), "~", getenv("HOME"));
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "db path :" << p;

    if (!exists(p)) {
        create_directories(p);
    }

    std::string dbFileName = p + "/" + device_name + "_" + std::to_string(max_CU) + ".cd.pad.txt";
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "db file name :" << dbFileName;
    return {dbFileName};
}

std::string genTempFilePath(std::string name) {
    std::string dir_path;
    std::string file_path;
    dir_path =
        (/*boost::filesystem::*/ temp_directory_path() + "/" +
                                 /*boost::filesystem::*/ unique_path(/*"amd-tmp-%%%%-%%%%-%%%%-%%%%"*/));

    create_directories(dir_path);

    file_path = dir_path + "/" + name; //(dir_path / name).string();

    if (!std::ofstream {file_path, std::ios_base::out | std::ios_base::in | std::ios_base::trunc}
            .good()) {
        AMD_THROW("Failed to create temp file: " + file_path);
    }
    return file_path;
}

void writeFile(const std::string& content, const std::string& name) {
    FILE* fd = std::fopen(name.c_str(), "w");

    if (std::fwrite(content.c_str(), 1, content.size(), fd) != content.size()) {
        std::fclose(fd);
        AMD_THROW("Failed to write to src file");
    }

    std::fclose(fd);
}

} // namespace saber
} // namespace anakin
