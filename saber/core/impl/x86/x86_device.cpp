#include "core/device.h"
namespace anakin{

namespace saber{

template <>
void Device<X86>::create_stream() {
    // todo
    //LOG(WARNING) << "X86 create_stream is not implemented";
    _data_stream.clear();
    _compute_stream.clear();
    for (int i = 0; i < _max_stream; i++) {
        typedef TargetWrapper<X86> API;
        typename API::stream_t stream_data;
        typename API::stream_t stream_compute;
        //cudaStreamNonBlocking
        API::create_stream_with_flag(&stream_data, 1);
        API::create_stream_with_flag(&stream_compute, 1);
        _data_stream.push_back(stream_data);
        _compute_stream.push_back(stream_compute);
    }
}

template <>
void Device<X86>::get_info() {
    // todo
    LOG(WARNING) << "X86 get_info is not implemented";
}

template void Device<X86>::get_info();
template void Device<X86>::create_stream();

} //namespace saber

} //namespace anakin
