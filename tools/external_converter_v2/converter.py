#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from config import *

def launch(config, graph):
    logger(verbose.WARNING).feed("anakin parser dash board will be launch in site: ")
    graph.run_with_server(config.ip, config.port)


class DeepLearningFramework(enum.Enum):
    """Anakin parser supported deep learning framework enum
    """
    caffe = 'CAFFE'
    fluid = 'FLUID'
    lego = 'LEGO'
    tensorflow = 'TENSORFLOW'
    onnx = 'ONNX'
    houyi = 'HOUYI'

    def __str__(self):
        return self.value


def parse_args():
    """parse command args
    """
    arg_parser = argparse.ArgumentParser('Anakin Parser')

    # common args
    arg_parser.add_argument(
        '--debug', type=str, help='debug')
    arg_parser.add_argument(
        '--framework', type=DeepLearningFramework, choices=list(DeepLearningFramework), help='input framework')
    arg_parser.add_argument(
        '--save_path', type=str, help='output save directory')
    arg_parser.add_argument(
        '--result_name', type=str, help='id of output filename')
    arg_parser.add_argument(
        '--open_launch_board', type=int, help='open Anakin net display board')
    arg_parser.add_argument(
        '--board_server_ip', type=str, help='display board server ip')
    arg_parser.add_argument(
        '--board_server_port', type=int, help='display board server port')
    arg_parser.add_argument(
        '--optimized_graph_enable', type=int, help='OptimizedGraph enable')
    arg_parser.add_argument(
        '--optimized_graph_path', type=str, help='OptimizedGraph path')
    arg_parser.add_argument(
        '--log_path', type=str, help='log dir')
    arg_parser.add_argument(
        '--log_with_color', type=str, help='use color log')

    # framwork specific args
    # CAFFE
    arg_parser.add_argument(
        '--caffe_proto_paths', nargs='*', help='caffe ProtoPaths')
    arg_parser.add_argument(
        '--caffe_proto_txt_path', type=str, help='caffe PrototxtPath')
    arg_parser.add_argument(
        '--caffe_model_path', type=str, help='caffe ModelPath')
    arg_parser.add_argument(
        '--caffe_remark', type=str, help='caffe Remark')

    # FLUID
    arg_parser.add_argument(
        '--fluid_debug', type=str, help='fluid debug switch')
    arg_parser.add_argument(
        '--fluid_model_path', type=str, help='fluid ModelPath')
    arg_parser.add_argument(
        '--fluid_net_type', type=str, help='fluid NetType')

    # LEGO
    arg_parser.add_argument(
        '--lego_proto_path', type=str, help='lego ProtoPath')
    arg_parser.add_argument(
        '--lego_prototxt_path', type=str, help='lego PrototxtPath')
    arg_parser.add_argument(
        '--lego_model_path', type=str, help='lego ModelPath')

    # TENSORFLOW
    arg_parser.add_argument(
        '--tensorflow_model_path', type=str, help='tensorflow ModelPath')
    arg_parser.add_argument(
        '--tensorflow_outputs', type=str, help='tensorflow OutPuts')

    # ONNX
    arg_parser.add_argument(
        '--onnx_model_path', type=str, help='onnx ModelPath')

    # HOUYI
    arg_parser.add_argument(
        '--houyi_model_path', type=str, help='houyi ModelPath')
    arg_parser.add_argument(
        '--houyi_weights_path', type=str, help='houyi WeightsPath')

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    config = Configuration(args)
    from parser import *
    logger.init(config.logger_dict)

    if config.DebugConfig is None:
        graph = Graph(config)
        graph.info_table()
        graph.serialization()

        if config.LaunchBoard:
            launch(config, graph)
    else:
        import utils
        net = utils.net.net_parser.NetHolder(config)
        net.parse()

