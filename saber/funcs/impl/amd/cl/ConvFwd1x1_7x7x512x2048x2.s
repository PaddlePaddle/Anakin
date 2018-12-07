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

.hsa_code_object_version 2, 1
.hsa_code_object_isa 8, 0, 3, "AMD", "AMDGPU"

.text
.globl ConvFwd1x1
.p2align 8
.type ConvFwd1x1,@function
.amdgpu_hsa_kernel ConvFwd1x1

ConvFwd1x1:
    .amd_kernel_code_t
        amd_code_version_major = 1
        amd_code_version_minor = 1
        amd_machine_kind = 1
        amd_machine_version_major = 8
        amd_machine_version_minor = 0
        amd_machine_version_stepping = 3
        kernarg_segment_alignment = 4
        group_segment_alignment = 4
        private_segment_alignment = 4
        wavefront_size = 6
        call_convention = -1
        enable_sgpr_kernarg_segment_ptr = 1
        enable_sgpr_workgroup_id_x = 1
        enable_sgpr_workgroup_id_y = 1
        enable_sgpr_workgroup_id_z = 1
        enable_vgpr_workitem_id = 2
        is_ptr64 = 1
        float_mode = 192
        granulated_wavefront_sgpr_count = 5
        granulated_workitem_vgpr_count = 7
        user_sgpr_count = 2
        wavefront_sgpr_count = 47
        workitem_vgpr_count = 32
        kernarg_segment_byte_size = 44
    .end_amd_kernel_code_t
    
START_PROG:
    s_load_dwordx2                              s[6:7], s[0:1], 0
    s_load_dwordx2                              s[8:9], s[0:1], 8
    s_load_dwordx2                              s[10:11], s[0:1], 16
    s_load_dwordx2                              s[12:13], s[0:1], 24
    s_load_dwordx2                              s[14:15], s[0:1], 32
    s_load_dword                                s[5], s[0:1], 40
    s_lshl_b32                                  s[20], s[2], 2                           
    v_lshrrev_b32                               v[16], 6, v[0]                           
    v_add_u32                                   v[2], vcc, v[16], s[20]                  
    v_and_b32                                   v[3], 63, v[0]                           
    v_lshrrev_b32                               v[17], 8, v[2]                           
    v_lshrrev_b32                               v[4], 0, v[17]                           
    v_and_b32                                   v[5], 0, v[17]                           
    v_and_b32                                   v[6], 255, v[2]                          
    v_lshlrev_b32                               v[16], 6, v[4]                           
    v_add_u32                                   v[16], vcc, v[3], v[16]                  
    v_mov_b32                                   v[17], 49
    v_cvt_f32_u32                               v[8], v[16]
    v_cvt_f32_u32                               v[7], v[17]
    v_rcp_f32                                   v[7], v[7]
    v_mul_f32                                   v[7], v[8], v[7]                         
    v_cvt_flr_i32_f32                           v[8], v[7]
    v_mul_u32_u24                               v[7], v[8], v[17]                        
    v_sub_u32                                   v[7], vcc, v[16], v[7]                   
    v_lshlrev_b32                               v[9], 3, v[6]                            
    v_mov_b32                                   v[16], 2
    v_cmpx_lt_u32                               vcc, v[8], v[16]                         
    s_cbranch_execz                             END_PROG
    v_mov_b32                                   v[16], 25088
    v_mul_u32_u24                               v[16], v[8], v[16]                       
    v_lshlrev_b32                               v[17], 9, v[5]                           
    v_mov_b32                                   v[18], 49
    v_mul_u32_u24                               v[17], v[17], v[18]                      
    v_add_u32                                   v[18], vcc, v[16], v[17]                 
    v_addc_u32                                  v[18], vcc, v[18], v[7], vcc
    v_lshlrev_b32                               v[18], 2, v[18]                          
    s_waitcnt                                   lgkmcnt(0)
    v_mov_b32                                   v[11], s[7]
    v_add_u32                                   v[10], vcc, s[6], v[18]                  
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    v_mov_b32                                   v[16], 512
    v_mul_u32_u24                               v[16], v[9], v[16]                       
    v_lshlrev_b32                               v[17], 9, v[5]                           
    v_add_u32                                   v[16], vcc, v[16], v[17]                 
    v_readfirstlane_b32                         s[20], v[16]
    s_lshl_b32                                  s[20], s[20], 2                          
    s_waitcnt                                   lgkmcnt(0)
    s_add_u32                                   s[16], s[8], s[20]                       
    s_addc_u32                                  s[17], 0, s[9]                           
    v_lshlrev_b32                               v[16], 2, v[9]                           
    v_mov_b32                                   v[13], s[11]
    v_add_u32                                   v[12], vcc, s[10], v[16]                 
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    v_mov_b32                                   v[17], 256
    v_mul_u32_u24                               v[16], v[4], v[17]                       
    v_add_u32                                   v[16], vcc, v[16], v[6]                  
    v_readfirstlane_b32                         s[20], v[16]
    s_lshl_b32                                  s[20], s[20], 2                          
    s_waitcnt                                   lgkmcnt(0)
    s_add_u32                                   s[18], s[14], s[20]                      
    s_addc_u32                                  s[19], 0, s[15]                          
    v_mov_b32                                   v[16], 100352
    v_mul_u32_u24                               v[16], v[8], v[16]                       
    v_mov_b32                                   v[17], 49
    v_mul_u32_u24                               v[17], v[9], v[17]                       
    v_add_u32                                   v[18], vcc, v[16], v[17]                 
    v_addc_u32                                  v[18], vcc, v[18], v[7], vcc
    v_lshlrev_b32                               v[18], 2, v[18]                          
    v_mov_b32                                   v[15], s[13]
    v_add_u32                                   v[14], vcc, s[12], v[18]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    v_mov_b32                                   v[24], 0
    v_mov_b32                                   v[25], 0
    v_mov_b32                                   v[26], 0
    v_mov_b32                                   v[27], 0
    v_mov_b32                                   v[28], 0
    v_mov_b32                                   v[29], 0
    v_mov_b32                                   v[30], 0
    v_mov_b32                                   v[31], 0
    flat_load_dword                             v[24], v[12:13]
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[25], v[12:13]
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[26], v[12:13]
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[27], v[12:13]
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[28], v[12:13]
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[29], v[12:13]
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[30], v[12:13]
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[31], v[12:13]
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    s_load_dword                                s[32], s[16:17], 0
    s_load_dword                                s[33], s[16:17], 2048
    s_load_dword                                s[34], s[16:17], 4096
    s_load_dword                                s[35], s[16:17], 6144
    s_load_dword                                s[36], s[16:17], 8192
    s_load_dword                                s[37], s[16:17], 10240
    s_load_dword                                s[38], s[16:17], 12288
    s_load_dword                                s[39], s[16:17], 14336
    flat_load_dword                             v[2], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[3], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[4], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[5], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[6], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[7], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[8], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[9], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_mov_b32                                   s[6], 31
CONV_LOOP:
    flat_load_dword                             v[16], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[17], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[18], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[19], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[20], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[21], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[22], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[23], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_load_dwordx8                              s[8:15], s[16:17], 0
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2048
    s_waitcnt                                   vmcnt(8)
    v_mac_f32                                   v[24], v[2], s[8]                        
    v_mac_f32                                   v[24], v[3], s[9]                        
    v_mac_f32                                   v[24], v[4], s[10]                       
    v_mac_f32                                   v[24], v[5], s[11]                       
    v_mac_f32                                   v[24], v[6], s[12]                       
    v_mac_f32                                   v[24], v[7], s[13]                       
    v_mac_f32                                   v[24], v[8], s[14]                       
    v_mac_f32                                   v[24], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 4096
    v_mac_f32                                   v[25], v[2], s[24]                       
    v_mac_f32                                   v[25], v[3], s[25]                       
    v_mac_f32                                   v[25], v[4], s[26]                       
    v_mac_f32                                   v[25], v[5], s[27]                       
    v_mac_f32                                   v[25], v[6], s[28]                       
    v_mac_f32                                   v[25], v[7], s[29]                       
    v_mac_f32                                   v[25], v[8], s[30]                       
    v_mac_f32                                   v[25], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 6144
    v_mac_f32                                   v[26], v[2], s[8]                        
    v_mac_f32                                   v[26], v[3], s[9]                        
    v_mac_f32                                   v[26], v[4], s[10]                       
    v_mac_f32                                   v[26], v[5], s[11]                       
    v_mac_f32                                   v[26], v[6], s[12]                       
    v_mac_f32                                   v[26], v[7], s[13]                       
    v_mac_f32                                   v[26], v[8], s[14]                       
    v_mac_f32                                   v[26], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 8192
    v_mac_f32                                   v[27], v[2], s[24]                       
    v_mac_f32                                   v[27], v[3], s[25]                       
    v_mac_f32                                   v[27], v[4], s[26]                       
    v_mac_f32                                   v[27], v[5], s[27]                       
    v_mac_f32                                   v[27], v[6], s[28]                       
    v_mac_f32                                   v[27], v[7], s[29]                       
    v_mac_f32                                   v[27], v[8], s[30]                       
    v_mac_f32                                   v[27], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 10240
    v_mac_f32                                   v[28], v[2], s[8]                        
    v_mac_f32                                   v[28], v[3], s[9]                        
    v_mac_f32                                   v[28], v[4], s[10]                       
    v_mac_f32                                   v[28], v[5], s[11]                       
    v_mac_f32                                   v[28], v[6], s[12]                       
    v_mac_f32                                   v[28], v[7], s[13]                       
    v_mac_f32                                   v[28], v[8], s[14]                       
    v_mac_f32                                   v[28], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 12288
    v_mac_f32                                   v[29], v[2], s[24]                       
    v_mac_f32                                   v[29], v[3], s[25]                       
    v_mac_f32                                   v[29], v[4], s[26]                       
    v_mac_f32                                   v[29], v[5], s[27]                       
    v_mac_f32                                   v[29], v[6], s[28]                       
    v_mac_f32                                   v[29], v[7], s[29]                       
    v_mac_f32                                   v[29], v[8], s[30]                       
    v_mac_f32                                   v[29], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 14336
    v_mac_f32                                   v[30], v[2], s[8]                        
    v_mac_f32                                   v[30], v[3], s[9]                        
    v_mac_f32                                   v[30], v[4], s[10]                       
    v_mac_f32                                   v[30], v[5], s[11]                       
    v_mac_f32                                   v[30], v[6], s[12]                       
    v_mac_f32                                   v[30], v[7], s[13]                       
    v_mac_f32                                   v[30], v[8], s[14]                       
    v_mac_f32                                   v[30], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    v_mac_f32                                   v[31], v[2], s[24]                       
    v_mac_f32                                   v[31], v[3], s[25]                       
    v_mac_f32                                   v[31], v[4], s[26]                       
    v_mac_f32                                   v[31], v[5], s[27]                       
    v_mac_f32                                   v[31], v[6], s[28]                       
    v_mac_f32                                   v[31], v[7], s[29]                       
    v_mac_f32                                   v[31], v[8], s[30]                       
    v_mac_f32                                   v[31], v[9], s[31]                       
    flat_load_dword                             v[2], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[3], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[4], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[5], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[6], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[7], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[8], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[9], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_load_dwordx8                              s[8:15], s[16:17], 32
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2080
    s_waitcnt                                   vmcnt(8)
    v_mac_f32                                   v[24], v[16], s[8]                       
    v_mac_f32                                   v[24], v[17], s[9]                       
    v_mac_f32                                   v[24], v[18], s[10]                      
    v_mac_f32                                   v[24], v[19], s[11]                      
    v_mac_f32                                   v[24], v[20], s[12]                      
    v_mac_f32                                   v[24], v[21], s[13]                      
    v_mac_f32                                   v[24], v[22], s[14]                      
    v_mac_f32                                   v[24], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 4128
    v_mac_f32                                   v[25], v[16], s[24]                      
    v_mac_f32                                   v[25], v[17], s[25]                      
    v_mac_f32                                   v[25], v[18], s[26]                      
    v_mac_f32                                   v[25], v[19], s[27]                      
    v_mac_f32                                   v[25], v[20], s[28]                      
    v_mac_f32                                   v[25], v[21], s[29]                      
    v_mac_f32                                   v[25], v[22], s[30]                      
    v_mac_f32                                   v[25], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 6176
    v_mac_f32                                   v[26], v[16], s[8]                       
    v_mac_f32                                   v[26], v[17], s[9]                       
    v_mac_f32                                   v[26], v[18], s[10]                      
    v_mac_f32                                   v[26], v[19], s[11]                      
    v_mac_f32                                   v[26], v[20], s[12]                      
    v_mac_f32                                   v[26], v[21], s[13]                      
    v_mac_f32                                   v[26], v[22], s[14]                      
    v_mac_f32                                   v[26], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 8224
    v_mac_f32                                   v[27], v[16], s[24]                      
    v_mac_f32                                   v[27], v[17], s[25]                      
    v_mac_f32                                   v[27], v[18], s[26]                      
    v_mac_f32                                   v[27], v[19], s[27]                      
    v_mac_f32                                   v[27], v[20], s[28]                      
    v_mac_f32                                   v[27], v[21], s[29]                      
    v_mac_f32                                   v[27], v[22], s[30]                      
    v_mac_f32                                   v[27], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 10272
    v_mac_f32                                   v[28], v[16], s[8]                       
    v_mac_f32                                   v[28], v[17], s[9]                       
    v_mac_f32                                   v[28], v[18], s[10]                      
    v_mac_f32                                   v[28], v[19], s[11]                      
    v_mac_f32                                   v[28], v[20], s[12]                      
    v_mac_f32                                   v[28], v[21], s[13]                      
    v_mac_f32                                   v[28], v[22], s[14]                      
    v_mac_f32                                   v[28], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 12320
    v_mac_f32                                   v[29], v[16], s[24]                      
    v_mac_f32                                   v[29], v[17], s[25]                      
    v_mac_f32                                   v[29], v[18], s[26]                      
    v_mac_f32                                   v[29], v[19], s[27]                      
    v_mac_f32                                   v[29], v[20], s[28]                      
    v_mac_f32                                   v[29], v[21], s[29]                      
    v_mac_f32                                   v[29], v[22], s[30]                      
    v_mac_f32                                   v[29], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 14368
    v_mac_f32                                   v[30], v[16], s[8]                       
    v_mac_f32                                   v[30], v[17], s[9]                       
    v_mac_f32                                   v[30], v[18], s[10]                      
    v_mac_f32                                   v[30], v[19], s[11]                      
    v_mac_f32                                   v[30], v[20], s[12]                      
    v_mac_f32                                   v[30], v[21], s[13]                      
    v_mac_f32                                   v[30], v[22], s[14]                      
    v_mac_f32                                   v[30], v[23], s[15]                      
    s_add_u32                                   s[16], s[16], 64                         
    s_addc_u32                                  s[17], s[17], 0                          
    s_waitcnt                                   lgkmcnt(0)
    s_load_dword                                s[32], s[16:17], 0
    s_load_dword                                s[33], s[16:17], 2048
    s_load_dword                                s[34], s[16:17], 4096
    s_load_dword                                s[35], s[16:17], 6144
    s_load_dword                                s[36], s[16:17], 8192
    s_load_dword                                s[37], s[16:17], 10240
    s_load_dword                                s[38], s[16:17], 12288
    s_load_dword                                s[39], s[16:17], 14336
    v_mac_f32                                   v[31], v[16], s[24]                      
    v_mac_f32                                   v[31], v[17], s[25]                      
    v_mac_f32                                   v[31], v[18], s[26]                      
    v_mac_f32                                   v[31], v[19], s[27]                      
    v_mac_f32                                   v[31], v[20], s[28]                      
    v_mac_f32                                   v[31], v[21], s[29]                      
    v_mac_f32                                   v[31], v[22], s[30]                      
    v_mac_f32                                   v[31], v[23], s[31]                      
    s_sub_u32                                   s[6], s[6], 1                            
    s_cmpk_eq_i32                               s[6], 0
    s_cbranch_scc0                              CONV_LOOP
    flat_load_dword                             v[16], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[17], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[18], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[19], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[20], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[21], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[22], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[23], v[10:11]
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_load_dwordx8                              s[8:15], s[16:17], 0
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2048
    s_waitcnt                                   vmcnt(8)
    v_mac_f32                                   v[24], v[2], s[8]                        
    v_mac_f32                                   v[24], v[3], s[9]                        
    v_mac_f32                                   v[24], v[4], s[10]                       
    v_mac_f32                                   v[24], v[5], s[11]                       
    v_mac_f32                                   v[24], v[6], s[12]                       
    v_mac_f32                                   v[24], v[7], s[13]                       
    v_mac_f32                                   v[24], v[8], s[14]                       
    v_mac_f32                                   v[24], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 4096
    v_mac_f32                                   v[25], v[2], s[24]                       
    v_mac_f32                                   v[25], v[3], s[25]                       
    v_mac_f32                                   v[25], v[4], s[26]                       
    v_mac_f32                                   v[25], v[5], s[27]                       
    v_mac_f32                                   v[25], v[6], s[28]                       
    v_mac_f32                                   v[25], v[7], s[29]                       
    v_mac_f32                                   v[25], v[8], s[30]                       
    v_mac_f32                                   v[25], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 6144
    v_mac_f32                                   v[26], v[2], s[8]                        
    v_mac_f32                                   v[26], v[3], s[9]                        
    v_mac_f32                                   v[26], v[4], s[10]                       
    v_mac_f32                                   v[26], v[5], s[11]                       
    v_mac_f32                                   v[26], v[6], s[12]                       
    v_mac_f32                                   v[26], v[7], s[13]                       
    v_mac_f32                                   v[26], v[8], s[14]                       
    v_mac_f32                                   v[26], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 8192
    v_mac_f32                                   v[27], v[2], s[24]                       
    v_mac_f32                                   v[27], v[3], s[25]                       
    v_mac_f32                                   v[27], v[4], s[26]                       
    v_mac_f32                                   v[27], v[5], s[27]                       
    v_mac_f32                                   v[27], v[6], s[28]                       
    v_mac_f32                                   v[27], v[7], s[29]                       
    v_mac_f32                                   v[27], v[8], s[30]                       
    v_mac_f32                                   v[27], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 10240
    v_mac_f32                                   v[28], v[2], s[8]                        
    v_mac_f32                                   v[28], v[3], s[9]                        
    v_mac_f32                                   v[28], v[4], s[10]                       
    v_mac_f32                                   v[28], v[5], s[11]                       
    v_mac_f32                                   v[28], v[6], s[12]                       
    v_mac_f32                                   v[28], v[7], s[13]                       
    v_mac_f32                                   v[28], v[8], s[14]                       
    v_mac_f32                                   v[28], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 12288
    v_mac_f32                                   v[29], v[2], s[24]                       
    v_mac_f32                                   v[29], v[3], s[25]                       
    v_mac_f32                                   v[29], v[4], s[26]                       
    v_mac_f32                                   v[29], v[5], s[27]                       
    v_mac_f32                                   v[29], v[6], s[28]                       
    v_mac_f32                                   v[29], v[7], s[29]                       
    v_mac_f32                                   v[29], v[8], s[30]                       
    v_mac_f32                                   v[29], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 14336
    v_mac_f32                                   v[30], v[2], s[8]                        
    v_mac_f32                                   v[30], v[3], s[9]                        
    v_mac_f32                                   v[30], v[4], s[10]                       
    v_mac_f32                                   v[30], v[5], s[11]                       
    v_mac_f32                                   v[30], v[6], s[12]                       
    v_mac_f32                                   v[30], v[7], s[13]                       
    v_mac_f32                                   v[30], v[8], s[14]                       
    v_mac_f32                                   v[30], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    v_mac_f32                                   v[31], v[2], s[24]                       
    v_mac_f32                                   v[31], v[3], s[25]                       
    v_mac_f32                                   v[31], v[4], s[26]                       
    v_mac_f32                                   v[31], v[5], s[27]                       
    v_mac_f32                                   v[31], v[6], s[28]                       
    v_mac_f32                                   v[31], v[7], s[29]                       
    v_mac_f32                                   v[31], v[8], s[30]                       
    v_mac_f32                                   v[31], v[9], s[31]                       
    s_load_dwordx8                              s[8:15], s[16:17], 32
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2080
    s_waitcnt                                   vmcnt(0)
    v_mac_f32                                   v[24], v[16], s[8]                       
    v_mac_f32                                   v[24], v[17], s[9]                       
    v_mac_f32                                   v[24], v[18], s[10]                      
    v_mac_f32                                   v[24], v[19], s[11]                      
    v_mac_f32                                   v[24], v[20], s[12]                      
    v_mac_f32                                   v[24], v[21], s[13]                      
    v_mac_f32                                   v[24], v[22], s[14]                      
    v_mac_f32                                   v[24], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 4128
    v_mac_f32                                   v[25], v[16], s[24]                      
    v_mac_f32                                   v[25], v[17], s[25]                      
    v_mac_f32                                   v[25], v[18], s[26]                      
    v_mac_f32                                   v[25], v[19], s[27]                      
    v_mac_f32                                   v[25], v[20], s[28]                      
    v_mac_f32                                   v[25], v[21], s[29]                      
    v_mac_f32                                   v[25], v[22], s[30]                      
    v_mac_f32                                   v[25], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 6176
    v_mac_f32                                   v[26], v[16], s[8]                       
    v_mac_f32                                   v[26], v[17], s[9]                       
    v_mac_f32                                   v[26], v[18], s[10]                      
    v_mac_f32                                   v[26], v[19], s[11]                      
    v_mac_f32                                   v[26], v[20], s[12]                      
    v_mac_f32                                   v[26], v[21], s[13]                      
    v_mac_f32                                   v[26], v[22], s[14]                      
    v_mac_f32                                   v[26], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 8224
    v_mac_f32                                   v[27], v[16], s[24]                      
    v_mac_f32                                   v[27], v[17], s[25]                      
    v_mac_f32                                   v[27], v[18], s[26]                      
    v_mac_f32                                   v[27], v[19], s[27]                      
    v_mac_f32                                   v[27], v[20], s[28]                      
    v_mac_f32                                   v[27], v[21], s[29]                      
    v_mac_f32                                   v[27], v[22], s[30]                      
    v_mac_f32                                   v[27], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 10272
    v_mac_f32                                   v[28], v[16], s[8]                       
    v_mac_f32                                   v[28], v[17], s[9]                       
    v_mac_f32                                   v[28], v[18], s[10]                      
    v_mac_f32                                   v[28], v[19], s[11]                      
    v_mac_f32                                   v[28], v[20], s[12]                      
    v_mac_f32                                   v[28], v[21], s[13]                      
    v_mac_f32                                   v[28], v[22], s[14]                      
    v_mac_f32                                   v[28], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 12320
    v_mac_f32                                   v[29], v[16], s[24]                      
    v_mac_f32                                   v[29], v[17], s[25]                      
    v_mac_f32                                   v[29], v[18], s[26]                      
    v_mac_f32                                   v[29], v[19], s[27]                      
    v_mac_f32                                   v[29], v[20], s[28]                      
    v_mac_f32                                   v[29], v[21], s[29]                      
    v_mac_f32                                   v[29], v[22], s[30]                      
    v_mac_f32                                   v[29], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 14368
    v_mac_f32                                   v[30], v[16], s[8]                       
    v_mac_f32                                   v[30], v[17], s[9]                       
    v_mac_f32                                   v[30], v[18], s[10]                      
    v_mac_f32                                   v[30], v[19], s[11]                      
    v_mac_f32                                   v[30], v[20], s[12]                      
    v_mac_f32                                   v[30], v[21], s[13]                      
    v_mac_f32                                   v[30], v[22], s[14]                      
    v_mac_f32                                   v[30], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    v_mac_f32                                   v[31], v[16], s[24]                      
    v_mac_f32                                   v[31], v[17], s[25]                      
    v_mac_f32                                   v[31], v[18], s[26]                      
    v_mac_f32                                   v[31], v[19], s[27]                      
    v_mac_f32                                   v[31], v[20], s[28]                      
    v_mac_f32                                   v[31], v[21], s[29]                      
    v_mac_f32                                   v[31], v[22], s[30]                      
    v_mac_f32                                   v[31], v[23], s[31]                      
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[24], 0                            
    v_mul_f32                                   v[24], v[24], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[24]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[25], 0                            
    v_mul_f32                                   v[25], v[25], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[25]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[26], 0                            
    v_mul_f32                                   v[26], v[26], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[26]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[27], 0                            
    v_mul_f32                                   v[27], v[27], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[27]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[28], 0                            
    v_mul_f32                                   v[28], v[28], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[28]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[29], 0                            
    v_mul_f32                                   v[29], v[29], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[29]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[30], 0                            
    v_mul_f32                                   v[30], v[30], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[30]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[31], 0                            
    v_mul_f32                                   v[31], v[31], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[31]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
END_PROG:
    s_endpgm

.amd_amdgpu_hsa_metadata
{ Version: [1, 0],
  Kernels :
    - { Name: ConvFwd1x1,
        SymbolName: ConvFwd1x1,
        Language: OpenCL C, LanguageVersion: [ 1, 2 ],
        Attrs: { ReqdWorkGroupSize: [ 256, 1, 1 ] }
        CodeProps: { KernargSegmentSize: 44, GroupSegmentFixedSize : 0, PrivateSegmentFixedSize : 0, KernargSegmentAlign : 8, WavefrontSize : 64, MaxFlatWorkGroupSize : 512 }
        Args:
        - { Name: d_in  , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
        - { Name: d_wei , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
        - { Name: d_bias , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
        - { Name: d_out , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global  }
        - { Name: d_sig , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global  }
        - { Name: d_nSlop , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
      }
}
.end_amd_amdgpu_hsa_metadata

