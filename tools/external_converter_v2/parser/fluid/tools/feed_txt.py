#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
A separate Fluid test file for feeding specific data.
"""

import sys
import numpy as np
import os
import paddle.fluid as fluid
from paddle.fluid import debugger
from paddle.fluid import core
from prettytable import PrettyTable
from operator import mul
import subprocess

GLB_model_path = '/path/to/fluid/inference/model/'
GLB_feed_example = {
    'var_name': 'data',
    'tensor_shape': [n, c, h, w],
    'txt_path': '/path/to/input/txt/',
}
GLB_feed_list = [GLB_feed_example]

# Do not modify 
GLB_arg_name = ''
GLB_batch_size = 1

def load_inference_model(model_path, exe):
    """
    """
    model_abs_path = os.path.join(model_path, 'model')
    param_abs_path = os.path.join(model_path, 'params')
    if os.path.exists(model_abs_path) and os.path.exists(param_abs_path):
        return fluid.io.load_inference_model(model_path, exe, 'model', 'params')
    else:
        return fluid.io.load_inference_model(model_path, exe)

def print_feed_fetch(block, feed_target_names, fetch_targets):
    """
    """
    tag_list = ["Index", "Name", "Shape", "Data Type", "Tensor Type"]
    def add_var_table(table, var, idx):
        table.add_row([idx, var.name, var.shape, str(var.dtype), str(var.type)])
    def feed_table(block, var_name_list):
        table = PrettyTable(tag_list)
        for var_name in var_name_list:
            idx = var_name_list.index(var_name)
            var = block.var(var_name)
            add_var_table(table, var, idx)
        return table
    def fetch_table(var_list):
        idx = 0
        table = PrettyTable(tag_list)
        for var in var_list:
            add_var_table(table, var, idx)
            idx = idx + 1
        return table
    print "\n", "=========== FEED TABLE ==========="
    print feed_table(block, feed_target_names)
    print "\n", "=========== FETCH TABLE ==========="
    print fetch_table(fetch_targets), "\n"

def add_feed_list(feed_list, fluid_feed_dict=None):
    """
    """
    if fluid_feed_dict is None:
        fluid_feed_dict = dict()
    def numpy_from_txt(txt_path,
                       tensor_shape=list(),
                       dtype=np.float32,
                       delimiter=None,
                       comments='#'):
        data = np.loadtxt(txt_path, dtype, comments, delimiter)
        data_size = np.size(data)
        tensor_size = reduce(mul, tensor_shape)
        assert (data_size == tensor_size), \
         "data size of txt (%d) must be equal to shape size (%d)." % (data_size, tensor_size)
        return np.reshape(data, tensor_shape)

    def add_feed_var(input_dict, fluid_feed_dict):
        var_name = input_dict['var_name']
        tensor_shape = input_dict['tensor_shape']
        txt_path = input_dict['txt_path']
        if 'data_type' in input_dict.keys():
            dtype = input_dict['data_type']
        else:
            dtype = np.float32
        if 'delimiter' in input_dict.keys():
            delim = input_dict['delimiter']
        else:
            delim = None
        fluid_feed_dict[var_name] = numpy_from_txt(txt_path, tensor_shape, dtype, delim)
        return fluid_feed_dict
    for input_dict in feed_list:
        add_feed_var(input_dict, fluid_feed_dict)
        return fluid_feed_dict

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

def fluid_inference_test(model_path, feed_list):
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
        print_feed_fetch(global_block, feed_target_names, fetch_targets)
        draw(global_block)
        feed_list = add_feed_list(feed_list)
        fetch_targets = fetch_tmp_vars(global_block, fetch_targets, [GLB_arg_name])
        results = exe.run(program=net_program,
                          feed=feed_list,
                          fetch_list=fetch_targets,
                          return_numpy=False)
        print_results(results, fetch_targets)

if __name__ == "__main__":
    if len(sys.argv) == 1 and GLB_model_path == '':
        raise NameError('Usage: python ./all_ones.py path/to/model arg_name batch_size')
    if len(sys.argv) > 1 and GLB_model_path == '':
        GLB_model_path = sys.argv[1]
    if len(sys.argv) > 2:
        GLB_arg_name = sys.argv[2]
    if len(sys.argv) > 3:
        GLB_batch_size = sys.argv[3]
    fluid_inference_test(GLB_model_path, GLB_feed_list)
