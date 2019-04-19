#include "anakin_config.h"
#ifndef USE_SGX
#include "saber/funcs/impl/x86/mkldnn_helper.h"

namespace anakin{
namespace saber{
    
mkldnn_mem_format get_mkldnn_format(LayoutType layout){
    switch (layout){
        case Layout_NCHW:
            return mkldnn_mem_format::nchw;
        case Layout_NCHW_C8R:
            return mkldnn_mem_format::nChw8c;
        default :
            return mkldnn_mem_format::nchw;
    }
}
mkldnn_mem_format get_mkldnn_format(LayoutType in_layout, LayoutType out_layout){
    if (in_layout == Layout_NCHW){
        switch (out_layout){
            case Layout_NCHW:
                return mkldnn_mem_format::oihw;
            case Layout_NCHW_C8R:
                return mkldnn_mem_format::Oihw8o;
            default:
                return mkldnn_mem_format::format_undef;
        }
      
    }
    if (in_layout == Layout_NCHW_C8R){
        switch (out_layout){
            case Layout_NCHW:
                return mkldnn_mem_format::oIhw8i;
            case Layout_NCHW_C8R:
                return mkldnn_mem_format::OIhw8i8o;
            default:
                return mkldnn_mem_format::format_undef;
        }
    }
    return  mkldnn_mem_format::format_undef;
}
mkldnn_mem_dtype get_mkldnn_dtype(DataType dtype){
    switch (dtype){
        case AK_FLOAT:
            return mkldnn_mem_dtype::f32;
        case AK_INT8:
            return mkldnn_mem_dtype::u8;
        default:
            return mkldnn_mem_dtype::f32;
    }
}
desc<mkldnn_mem> create_mkldnn_memory_desc(
                    const std::vector<int>& dims,
                    mkldnn_mem_dtype dtype, 
                    mkldnn_mem_format layout){
  mkldnn_mem_dim tz = dims;
  return desc<mkldnn_mem>({tz}, dtype, layout);
}

mkldnn_mem_ptr create_mkldnn_memory(Tensor<X86>* tensor, mkldnn::engine e){

  mkldnn_mem_format mft = get_mkldnn_format(tensor -> get_layout());
  mkldnn_mem_dtype  dt = get_mkldnn_dtype(tensor -> get_dtype());
  mkldnn_mem_dim dim = tensor -> shape();
  
  return mkldnn_mem_ptr(new mkldnn_mem({ { {dim}, dt, mft}, e}, tensor->mutable_data())); 	
}
mkldnn_mem_ptr create_mkldnn_memory_no_data(const Tensor<X86>* tensor, mkldnn::engine e){

  mkldnn_mem_format mft = get_mkldnn_format(tensor -> get_layout());
  mkldnn_mem_dtype  dt = get_mkldnn_dtype(tensor -> get_dtype());
  mkldnn_mem_dim dim = tensor -> shape();
  
  return mkldnn_mem_ptr(new mkldnn_mem({ { {dim}, dt, mft}, e}));   
}
mkldnn_mem_ptr create_mkldnn_memory(Tensor<X86>* tensor, const std::vector<int>& sh, mkldnn::engine e){
  mkldnn_mem_format mft = get_mkldnn_format(tensor -> get_layout());
  mkldnn_mem_dtype  dt = get_mkldnn_dtype(tensor -> get_dtype());
  mkldnn_mem_dim dim = sh;
  
  return mkldnn_mem_ptr(new mkldnn_mem({ { {dim}, dt, mft}, e}, tensor->mutable_data())); 	
}

mkldnn_mem_ptr create_mkldnn_memory(Tensor<X86>* tensor,const std::vector<int>& sh, 
    mkldnn_mem_format mft, mkldnn_mem_dtype dt, mkldnn::engine e){
	mkldnn_mem_dim dim = sh;
  return  mkldnn_mem_ptr(new mkldnn_mem({ { {dim}, dt, mft}, e}, tensor->mutable_data())); 	
}


}
}
#endif
