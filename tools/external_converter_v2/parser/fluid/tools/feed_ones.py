#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
A separate Fluid test file for feeding specific data.
'''

import sys
import numpy as np
import os
import paddle.fluid as fluid
from paddle.fluid import debugger
from paddle.fluid import core
import subprocess

GLB_model_path = ''
GLB_arg_name = ''
GLB_batch_size = 1

def load_inference_model(model_path, exe):
    '''
    '''
    model_abs_path = os.path.join(model_path, 'model')
    param_abs_path = os.path.join(model_path, 'params')
    if os.path.exists(model_abs_path) and os.path.exists(param_abs_path):
        return fluid.io.load_inference_model(model_path, exe, 'model', 'params')
    else:
        return fluid.io.load_inference_model(model_path, exe)

def feed_ones(block, feed_target_names, batch_size=1):
    """ 
    """ 
    feed_dict = dict()
    def set_batch_size(shape, batch_size):
        if shape[0] == -1:
            shape[0] = batch_size
        return shape
    def fill_ones(var_name, batch_size):
        var = block.var(var_name)
        np_shape = set_batch_size(list(var.shape), 1)
        var_np = {
            core.VarDesc.VarType.BOOL: np.bool_,
            core.VarDesc.VarType.INT32: np.int32,
            core.VarDesc.VarType.INT64: np.int64,
            core.VarDesc.VarType.FP16: np.float16,
            core.VarDesc.VarType.FP32: np.float32,
            core.VarDesc.VarType.FP64: np.float64,
        }
        np_dtype = var_np[var.dtype]
        return np.ones(np_shape, dtype=np_dtype)
    for feed_target_name in feed_target_names:
        feed_dict[feed_target_name] = fill_ones(feed_target_name, batch_size)
    return feed_dict

def draw(block, filename='debug'):
    """
    """
    dot_path = './' + filename + '.dot'
    pdf_path = './' + filename + '.pdf'
    debugger.draw_block_graphviz(block, path=dot_path)
    cmd = ["dot", "-Tpdf", dot_path, "-o", pdf_path]
    subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def fetch_tmp_vars(block, fetch_targets, var_names_list=None):
    """
    """
    def var_names_of_fetch(fetch_targets):
        var_names_list = []
        for var in fetch_targets:
            var_names_list.append(var.name)
        return var_names_list

    fetch_var = block.var('fetch')
    old_fetch_names = var_names_of_fetch(fetch_targets)
    new_fetch_vars = []
    for var_name in old_fetch_names:
        var = block.var(var_name)
        new_fetch_vars.append(var)
    i = len(new_fetch_vars)
    if var_names_list is None:
        var_names_list = block.vars.keys()
    for var_name in var_names_list:
        if '.tmp_' in var_name and var_name not in old_fetch_names:
            var = block.var(var_name)
            new_fetch_vars.append(var)
            block.append_op(
                type='fetch',
                inputs={'X': [var_name]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})
            i = i + 1
    return new_fetch_vars

def print_ops_type(block):
    """
    """
    def ops_type(block):
        ops = list(block.ops)
        cache = []
        for op in ops:
            if op.type not in cache:
                cache.append(op.type)
        return cache
    type_cache = ops_type(block)
    print 'type: '
    for op_type in type_cache:
        print op_type

def print_results(results, fetch_targets, need_save=True):
    """
    """
    for result in results:
        idx = results.index(result)
        print fetch_targets[idx]
        print np.array(result)
        if need_save is True:
            fluid_fetch_list = list(np.array(result).flatten())
            fetch_txt_fp = open('result_' + fetch_targets[idx].name + '.txt', 'w')
            for num in fluid_fetch_list:
                fetch_txt_fp.write(str(num) + '\n')
            fetch_txt_fp.close()

def fluid_inference_test(model_path):
    """
    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        [net_program, 
        feed_target_names, 
        fetch_targets] = load_inference_model(model_path, exe)
        global_block = net_program.global_block()
        draw(global_block)
        feed_list = feed_ones(global_block, feed_target_names)
        fetch_targets = fetch_tmp_vars(global_block, fetch_targets, [GLB_arg_name])
        results = exe.run(program=net_program,
                          feed=feed_list,
                          fetch_list=fetch_targets,
                          return_numpy=False)
        print_results(results, fetch_targets)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise NameError('Usage: python ./all_ones.py path/to/model arg_name batch_size')
    if len(sys.argv) > 1:
        GLB_model_path = sys.argv[1]
    if len(sys.argv) > 2:
        GLB_arg_name = sys.argv[2]
    if len(sys.argv) > 3:
        GLB_batch_size = sys.argv[3]
    fluid_inference_test(GLB_model_path)

