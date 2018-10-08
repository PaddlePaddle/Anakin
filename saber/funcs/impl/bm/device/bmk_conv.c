#include <stdio.h>
#include "bm_common.h"
#include "atomic_dma_gen_cmd.h"
#include "atomic_conv_gen_cmd.h"

int bm_conv_fwd(bm_api_conv_forward conv_param)
{
    // Unpack parameters
    u64 ifmap_offset_global = conv_param.ifmap_offset_global;
    u64 ofmap_offset_global = conv_param.ofmap_offset_global;
    u64 weight_offset_global = conv_param.weight_offset_global;
    u64 bias_offset_global = conv_param.bias_offset_global;
    int input_n = conv_param.input_n;
    int input_c = conv_param.input_c;
    int input_h = conv_param.input_h;
    int input_w = conv_param.input_w;
    int groups = conv_param.groups;
    int output_c = conv_param.output_c;
    int kh = conv_param.kh;
    int kw = conv_param.kw;
    int dh = conv_param.dh;
    int dw = conv_param.dw;
    int pad_h = conv_param.pad_h;
    int pad_w = conv_param.pad_w;
    int stride_h = conv_param.stride_h;
    int stride_w = conv_param.stride_w;
    int using_bias = conv_param.using_bias;
    int result_add = conv_param.result_add;
    int icsecs = conv_param.icsecs;
    int ocsecs = conv_param.ocsecs;
    int nsecs = conv_param.nsecs;
    int hsecs = conv_param.hsecs;

    P_COMMAND dma_command;
    CMD_ID_NODE id_node;
    resync_cmd_id( &id_node );

    int kh_ext = dh * (kh - 1) + 1;
    int kw_ext = dw * (kw - 1) + 1;
    int output_h = (input_h + 2 * pad_h - kh_ext) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kw_ext) / stride_w + 1;

    int ic = input_c / groups;
    int oc = output_c / groups;
    int ic_per_NPU = ceiling_func_shift(ic, NPU_SHIFT);
    int oc_per_NPU = ceiling_func_shift(oc, NPU_SHIFT);
    int bias_offset_local = 0;
    int bias_tensor_size = oc_per_NPU * FLOAT_SIZE;
    int weight_offset_local = bias_offset_local + bias_tensor_size;
    int weight_group_offset = oc * ic * kh * kw;
    int weight_tensor_size = ic * oc_per_NPU * kh * kw * FLOAT_SIZE;
    int weight_capacity = addr_EU_align(weight_tensor_size + bias_tensor_size);
    int ifmap_group_offset = ic * input_h * input_w;
    int ofmap_group_offset = oc * output_h * output_w;
    int global_ifmap_Nstride = ifmap_group_offset * groups;
    int global_ofmap_Nstride = ofmap_group_offset * groups;
    int nslice = input_n, ocslice = oc, icslice = ic, hslice = output_h;
    nslice = input_n / nsecs;
    int n_residual = input_n - nslice * nsecs;
    hslice = output_h / hsecs;
    int h_residual = output_h - hslice * hsecs;
    icslice = ic / icsecs;
    int ic_residual = ic - icslice * icsecs;
    ocslice = oc / ocsecs;
    int oc_residual = oc - ocslice * ocsecs;
    int bias_group_offset = oc;
    int max_icslice = icslice + (ic_residual > 0);
    int max_ic_per_NPU = ceiling_func_shift(max_icslice, NPU_SHIFT);
    int max_ocslice = ocslice + (oc_residual > 0);
    int max_oc_per_NPU = ceiling_func_shift(max_ocslice, NPU_SHIFT);

    for (int ig = 0; ig < groups; ig++){
        int ocend = 0;
        for (int ocidx = 0; ocidx < ocsecs; ocidx++){
            int ocstart = ocend;
            int cur_ocslice = ocslice + (oc_residual > ocidx);
            ocend = ocstart + cur_ocslice;
            oc_per_NPU = ceiling_func_shift(cur_ocslice, NPU_SHIFT);
            if (using_bias){
                dma_command = get_command(ENGINE_GDMA);
                tensor_compact_move_gen_cmd(
                    bias_offset_local, // local mem start address
                    bias_offset_global + (ig * bias_group_offset + ocstart) * FLOAT_SIZE, // global mem start address
                    1, cur_ocslice, 1, 1, // n, c, h, w
                    0, // direction G2L
                    0, // transpose
                    (void *)dma_command,
                    0, // local mem index
                    &id_node
                );
                call_atomic(nodechip_idx, atomic_global_dma, dma_command, ENGINE_GDMA);
            }
            weight_capacity = max_icslice * oc_per_NPU * kh * kw * FLOAT_SIZE;
            int ofmap_offset_local = addr_EU_align(weight_capacity + weight_offset_local);
            int nend = 0;
            for (int nidx = 0; nidx < nsecs; nidx++){
                int nstart = nend;
                int sec_len_n = nslice + (nidx < n_residual);
                nend = nstart + sec_len_n;
                int o_hb = 0;
                for (int hidx = 0; hidx < hsecs; hidx++){
                    int o_ht = o_hb;
                    int o_h = hslice + (h_residual > hidx);
                    o_hb = o_ht + o_h;
                    int i_ht = bm_max(o_ht * stride_h - pad_h, 0);
                    int pad_h_t = 0;
                    if (i_ht == 0){
                        pad_h_t = pad_h - o_ht * stride_h;
                    }
                    int i_hb = bm_min(o_hb * stride_h + kh_ext - 1 - pad_h, input_h);
                    int pad_h_b = 0;
                    if (i_hb == input_h){
                        pad_h_b = o_hb * stride_h + kh_ext - 1 - pad_h - input_h;
                    }
                    int i_h = i_hb - i_ht;
                    int ifmap_align_size = get_neuron_csize_local(i_h, input_w);
                    int ifmap_tensor_size = sec_len_n * max_ic_per_NPU * ifmap_align_size;
                    int ofmap_align_size = get_neuron_csize_local(o_h, output_w);
                    int ofmap_tensor_size = sec_len_n * max_oc_per_NPU * ofmap_align_size;
                    int ifmap_offset_local = ofmap_offset_local + ofmap_tensor_size;
                    int offset_local_end = ifmap_offset_local + ifmap_tensor_size;
                    ASSERT(offset_local_end <= LOCAL_MEM_SIZE);
                    if (result_add){
                        dma_command = get_command(ENGINE_GDMA);
                        u64 shift = nstart * global_ofmap_Nstride + ig * ofmap_group_offset +
                            (ocstart * output_h + o_ht) * output_w;
                        int local_cstride = get_cstride_local(o_h, output_w);
                        tensor_stride_move_gen_cmd(
                            ofmap_offset_local, // local mem start address
                            ofmap_offset_global + shift * FLOAT_SIZE, // global mem start address
                            sec_len_n, cur_ocslice, o_h, output_w, // n, c, h, w
                            0, // local mem index
                            0, // direction G2L
                            global_ofmap_Nstride, output_h * output_w, output_w, // src stride n,c,h
                            oc_per_NPU * local_cstride, local_cstride, output_w, // dst stride n,c,h
                            GDMA_TYPE_f32, 
                            0, // transpose 
                            dma_command, &id_node
                        );
                        call_atomic(nodechip_idx, atomic_global_dma, dma_command, ENGINE_GDMA);
                    }
                    int icend = 0;
                    for (int icidx = 0; icidx < icsecs; icidx++){
                        int icstart = icend;
                        int cur_icslice = icslice + (ic_residual > icidx);
                        icend = icstart + cur_icslice;
                        ic_per_NPU = ceiling_func_shift(cur_icslice, NPU_SHIFT);
                        u64 shift = (ocstart * ic + icstart) * kh * kw + ig * weight_group_offset;
                        if ((icsecs != 1) || (nidx == 0 && hidx == 0)){
                            dma_command = get_command(ENGINE_GDMA);
                            tensor_stride_move_gen_cmd(
                                weight_offset_local, // local mem start address
                                weight_offset_global + shift * FLOAT_SIZE, // global mem start address
                                1, cur_ocslice, cur_icslice, kh * kw, // n, c, h, w
                                0, // local mem index
                                0, // direction G2L
                                0, ic * kh * kw, kh * kw, // src stride n,c,h
                                0, cur_icslice * kh * kw, kh * kw, // dst stride n,c,h
                                GDMA_TYPE_f32, 
                                0, // transpose 
                                dma_command, &id_node
                            );
                            call_atomic(nodechip_idx, atomic_global_dma, dma_command, ENGINE_GDMA);
                        }
                        shift = nstart * global_ifmap_Nstride + ig * ifmap_group_offset +
                                (icstart * input_h + i_ht) * input_w;
                        int local_cstride = get_cstride_local(i_h, input_w);
                        dma_command = (float*)get_command(ENGINE_GDMA);
                        tensor_stride_move_gen_cmd(
                            ifmap_offset_local, // local mem start address
                            ifmap_offset_global + shift * FLOAT_SIZE, // global mem start address
                            sec_len_n, cur_icslice, i_h, input_w, // n, c, h, w
                            0, // local mem index
                            0, // direction G2L
                            global_ifmap_Nstride, input_h * input_w, input_w, // src stride n,c,h
                            ic_per_NPU * local_cstride, local_cstride, input_w, // dst stride n,c,h
                            GDMA_TYPE_f32, 
                            0, // transpose
                            dma_command, &id_node
                        );
                        call_atomic(nodechip_idx, atomic_global_dma, dma_command, ENGINE_GDMA);

                        local_shape_t ifshape, ofshape;
                        ifshape.n = sec_len_n;
                        ifshape.c = cur_icslice;
                        ifshape.h = i_h;
                        ifshape.w = input_w;
                        ofshape.c = cur_ocslice;
                        ofshape.h = o_h;
                        ofshape.w = output_w;
                        P_COMMAND conv_command = get_command(ENGINE_BD);
                        atomic_conv_kernel_stride_gen_cmd(
                            conv_command,
                            LOCAL_MEM_START_ADDR | ifmap_offset_local, // input address
                            LOCAL_MEM_START_ADDR | ofmap_offset_local, // output address
                            LOCAL_MEM_START_ADDR | weight_offset_local, // weight address
                            LOCAL_MEM_START_ADDR | bias_offset_local, // bias address
                            ifshape, // input shape
                            ofshape, // output shape
                            kh, kw, // kernel h, w
                            dh, dw, // dilation h, w
                            kh * kw, cur_icslice * kh * kw, kw, // kernel stride n,c,h
                            pad_h_t, pad_h_b, pad_w, pad_w, // pad top, bottom, left, right
                            stride_h, stride_w, // stride h, w
                            icidx == icsecs - 1 ? using_bias: 0, // use bias
                            result_add || icidx > 0, // add result
                            &id_node
                        );
                        call_atomic(nodechip_idx, atomic_conv_neuron, conv_command, ENGINE_BD);
                    }
                    u64 shift = nstart * global_ofmap_Nstride + ig * ofmap_group_offset +
                                (ocstart * output_h + o_ht) * output_w;
                    int local_cstride = get_cstride_local(o_h, output_w);
                    
                    dma_command = get_command(ENGINE_GDMA);
                    tensor_stride_move_gen_cmd(
                        ofmap_offset_local, // local mem start address
                        ofmap_offset_global + shift * FLOAT_SIZE, // global mem start address
                        sec_len_n, cur_ocslice, o_h, output_w, // n, c, h, w
                        0, // local mem index
                        1, // direction L2G
                        oc_per_NPU * local_cstride, local_cstride, output_w, // src stride n,c,h
                        global_ofmap_Nstride, output_h * output_w, output_w, // dst stride n,c,h
                        GDMA_TYPE_f32, 
                        0, // transpose
                        dma_command, &id_node
                    );
                    call_atomic(nodechip_idx, atomic_global_dma, dma_command, ENGINE_GDMA);
                }
            }
        }
    }
    poll_all_engine_done(&id_node);
    return 0;
}