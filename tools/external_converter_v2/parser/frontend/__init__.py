#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

from ..proto import *
from ..graph_to_json import GraphToJson
from ..graph_io import GraphProtoIO
from dash_board import GraphBoard


def RunServerOnGraph(f_graph_get):
    """
    launch dash board server
    """
    def warpper(self, *args):
        graph, config = f_graph_get(self, *args)
        graph_to_json, attrs, mem_info_hold= GraphToJson(graph)()
        # parser origin graph
        GraphBoard.config['graph_attrs'] = attrs
        GraphBoard.config['graph_option'] = graph_to_json
        GraphBoard.config['config'] = config

        # parser optimized graph from ankin
        graph_optimized = GraphProtoIO()
        if config.hasOptimizedGraph:
            graph_optimized.parse_from_string(config.optimizedGraphPath)
            optimized_graph_to_json, optimized_graph_attrs, mem_info= GraphToJson(graph_optimized)()
            GraphBoard.config['optimized_graph_attrs'] = optimized_graph_attrs
            GraphBoard.config['optimized_graph_option'] = optimized_graph_to_json 
            GraphBoard.config['mem_info'] = mem_info 
            GraphBoard.config['disable_optimization'] = False
        else:
            GraphBoard.config['disable_optimization'] = True

        # run server
        GraphBoard.run(host=args[0], port=args[1], debug=True, use_reloader=False)
    return warpper

__all__ = ["main", "RunServerOnGraph"]
