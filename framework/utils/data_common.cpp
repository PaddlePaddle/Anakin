#include "framework/utils/data_common.h"
namespace anakin {

std::vector<std::string> string_split(std::string in_str, std::string delimiter) {
    std::vector<std::string> seq;
    int found = in_str.find(delimiter);
    int pre_found = -1;
    while (found != std::string::npos) {
        if (pre_found == -1) {
            seq.push_back(in_str.substr(0, found));
        } else {
            seq.push_back(in_str.substr(pre_found + delimiter.length(), found - delimiter.length() - pre_found));
        }
        pre_found = found;
        found = in_str.find(delimiter, pre_found + delimiter.length());
    }
    seq.push_back(in_str.substr(pre_found+1, in_str.length() - (pre_found+1)));
    return seq;
}
std::vector<std::string> string_split(std::string in_str, std::vector<std::string> &delimiter) {
    std::vector<std::string> in;
    std::vector<std::string> out;
    out.push_back(in_str);
    for (auto del : delimiter) {
        in = out;
        out.clear();
        for (auto s : in) {
            auto out_s = string_split(s, del);
            for (auto o : out_s) {
                out.push_back(o);
            }
        }
    }
    return out;
}   
}   
