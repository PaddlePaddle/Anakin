#include "core/device.h"
namespace anakin{

namespace saber{

template <>
void Device<BM>::create_stream() {
    // todo
    LOG(WARNING) << "BM create_stream is not implemented";
}

template <>
void Device<BM>::get_info() {
    // todo
    LOG(WARNING) << "BM get_info is not implemented";
}

template void Device<BM>::get_info();
template void Device<BM>::create_stream();


} //namespace saber

} //namespace anakin
