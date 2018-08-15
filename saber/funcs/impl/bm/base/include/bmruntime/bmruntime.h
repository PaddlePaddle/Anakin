#ifndef  BMRUNTIME_H_
#define  BMRUNTIME_H_
#include <algorithm>
#include <vector>
#include "bmlib_runtime.h"
#include "bmruntime_common.h"
#include "stdio.h"
#include <string>
#include <map>
#include <set>
#include <iostream>

using std::vector;
using std::map;
using std::set;
using std::string;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;
typedef unsigned int            u32;
typedef unsigned long long      u64;

typedef struct stage_param_with_idx{
  int height_high;
  int height_low;
  int width_high;
  int width_low;
  int stage_index;
}stage_param_with_idx_t;

class bmruntime {
  public:
    bmruntime(bm_handle_t bm_handle);
    ~bmruntime();

    bool load_context(const string& ctx_dir);

    const set<string>& get_input_tensor(int net_idx) const;
    const set<string>& get_input_tensor(const string& net_name);

    const set<string>& get_output_tensor(int net_idx) const;
    const set<string>& get_output_tensor(const string& net_name);

    const bm_device_mem_t* get_input_blob(const string& tensor_name, int net_idx);
    const bm_device_mem_t* get_input_blob(const string& tensor_name, const string& net_name);

    const bm_device_mem_t* get_output_blob(const string& tensor_name, int net_idx);
    const bm_device_mem_t* get_output_blob(const string& tensor_name, const string& net_name);

    bool launch(int net_idx);
    bool launch(const string& net_name);

    bool launch(int net_idx, const bm_device_mem_t* input_tensors, int input_num,
            const bm_device_mem_t* output_tensors, int output_num);
    bool launch(const string& net_name, const bm_device_mem_t* input_tensors, int input_num,
            const bm_device_mem_t* output_tensors, int output_num);

    bool launch(int net_idx, int n, int h , int w);
    bool launch(const string& net_name, int n, int h, int w);
    bool launch(int net_idx, const bm_device_mem_t* input_tensors, int input_num,
            const bm_device_mem_t* output_tensors, int output_num, int n, int h, int w);
    bool launch(const string& net_name, const bm_device_mem_t* input_tensors, int input_num,
            const bm_device_mem_t* output_tensors, int output_num, int n , int h, int w);

    void get_input_blob_max_nhw(const string& tensor_name, int net_idx, int * max_n, int * max_c, int * max_h, int * max_w);
    void get_input_blob_max_nhw(const string& tensor_name, const string& net_name, int * max_n, int * max_c, int * max_h, int * max_w);
    void get_output_blob_max_nhw(const string& tensor_name, int net_idx, int * max_n, int * max_c, int * max_h, int * max_w);
    void get_output_blob_max_nhw(const string& tensor_name, const string& net_name, int * max_n, int *max_c, int * max_h, int * max_w);

    int get_oh_from_ih(const string& input_tensor_name, const string& output_tensor_name, const string& net_name, int ih);
    int get_oh_from_ih(const string& input_tensor_name, const string& output_tensor_name, int net_idx, int ih);
    int get_ow_from_iw(const string& input_tensor_name, const string& output_tensor_name, const string& net_name, int iw);
    int get_ow_from_iw(const string& input_tensor_name, const string& output_tensor_name, int net_idx, int iw);




    bool can_batch_size_change(int net_idx);
    bool can_batch_size_change(const string& net_name);
    bool can_height_and_width_change(int net_idx);
    bool can_height_and_width_change(const string& net_name);

    void show_neuron_network();

    int get_network_number() {return net_num;}

    inline bm_handle_t get_bm_handle() {return m_handle;}

  protected:
    bool setup_mem_context(const string& ctx_dir);
    bool setup_cmd_context(const string& ctx_dir);
    bool set_using_cmd_file(const string& ctx_dir);
    void load_cmd(u32* cmd, int engine_id, bool last_cmd, u64 start_address, u64 append_mem_offset);
    bool setup_ir_context(const string& ctx_dir);

    void wrong_net_idx_handle(int net_idx) const;

    int get_net_idx(const string& net_name);
    int get_stage_idx(int net_idx, int h, int w);
    u64 get_stage_offset(int net_idx, int stage_idx);

    int compute_output_height(int input_height, int global_kh, int global_stride_h, int global_pad_h, int global_pool_kh);
    int compute_output_width(int input_width, int global_kw, int global_stride_w, int global_pad_w, int global_pool_kw);

    bm_handle_t m_handle;
    std::vector<DEVICE_MEM_INFO>            m_device_mem_info_vec;
    std::vector<bm_device_mem_t>            m_device_mem_vec;

    vector<int>                             m_gdma_total_id_v;
    vector<int>                             m_cdma_total_id_v;
    vector<int>                             m_bdc_total_id_v;
    vector<vector<int> >                    m_gdma_group_id_v;
    vector<vector<int> >                    m_cdma_group_id_v;
    vector<vector<int> >                    m_bdc_group_id_v;
    vector<int>                             m_cmdgroup_num;
    vector<u64>                             m_gdma_cmd_start_address_v;
    vector<u64>                             m_cdma_cmd_start_address_v;
    vector<u64>                             m_bdc_cmd_start_address_v;
    vector<map<string, bm_device_mem_t> >   input_tensor_mem_map_v;
    vector<map<string, bm_device_mem_t> >   output_tensor_mem_map_v;
    vector<set<string> >                    m_input_tensor_set_v;
    vector<set<string> >                    m_output_tensor_set_v;
    int                                     net_num;
    map<string,int>                         net_name_to_idx;
    vector<int>                             stage_num;

    bool                                    have_ir_info;
    vector<vector<unsigned int> >           m_ir_info_len;
    vector<u64>                             m_ir_info_start_address_v;
    vector<vector<stage_param_with_idx_t> > stage_param_with_idx_vv;

    //io tensor param
    vector<int>                             n_can_change_v;
    vector<int>                             h_w_can_change_v;

    vector<vector<map<string, tensor_max_shape_t> > >           input_tensor_max_shape_vv;
    vector<vector<map<string, tensor_max_shape_t> > >           output_tensor_max_shape_vv;
    vector<vector<map<string, global_output_tensor_param_t> > > global_output_tensor_param_vv;

    bool m_using_cmd_file;
    FILE * m_gdma_cmd_file;
    FILE * m_cdma_cmd_file;
    FILE * m_bdc_cmd_file;

    //previous value or state
    int pre_net_num;
    int pre_m_device_mem_info_vec_size;  
    int pre_m_device_mem_vec_size;  

    //append mem offset when appending another framework's context.
    vector<u64> apd_ctx_mem_offset;
};

#endif
